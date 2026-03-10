import logging
import re
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class DownloadManager:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.queue = deque()
        self.active = {}
        self.max_workers = 1
        self.lock = threading.Lock()

    def add(self, url):
        job_id = str(uuid.uuid4())
        filename = url.split("/")[-1]
        if "?" in filename:
            filename = filename.split("?")[0]
        job = {
            "id": job_id,
            "url": url,
            "filename": filename,
            "progress": 0.0,
            "downloaded": 0,
            "total": 0,
            "speed": 0,
            "eta": 0,
            "status": "queued",
            "error": None,
            "cancel_flag": False,
        }
        with self.lock:
            self.queue.append(job)
        self._start_worker_if_possible()
        return job_id

    def _start_worker_if_possible(self):
        with self.lock:
            if len(self.active) >= self.max_workers or not self.queue:
                return
            job = self.queue.popleft()
            self.active[job["id"]] = job
        thread = threading.Thread(target=self._worker, args=(job,))
        thread.daemon = True
        thread.start()

    def _worker(self, job):
        job["status"] = "downloading"
        save_path = self.models_dir / job["filename"]
        downloaded = 0
        headers = {}
        if save_path.exists():
            downloaded = save_path.stat().st_size
            headers["Range"] = f"bytes={downloaded}-"
        start_time = time.time()
        try:
            with requests.get(job["url"], stream=True, headers=headers, timeout=10) as r:
                if r.status_code == 416:
                    job["progress"], job["status"] = 1.0, "done"
                    return
                if r.status_code not in (200, 206):
                    raise Exception(f"HTTP Error {r.status_code}")
                total = int(r.headers.get("Content-Length", 0)) + downloaded
                job["total"] = total
                mode = "ab" if downloaded else "wb"
                with open(save_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if job["cancel_flag"]:
                            job["status"] = "cancelled"
                            return
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            job["downloaded"] = downloaded
                            if total > 0:
                                job["progress"] = downloaded / total
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                job["speed"] = (
                                    downloaded
                                    - int(headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
                                ) / elapsed
                                remaining = total - downloaded
                                job["eta"] = remaining / job["speed"] if job["speed"] > 0 else 0
            job["progress"], job["status"] = 1.0, "done"
        except Exception as e:
            job["error"], job["status"] = str(e), "error"
        finally:
            with self.lock:
                self.active.pop(job["id"], None)
            self._start_worker_if_possible()

    def cancel(self, job_id):
        with self.lock:
            if job_id in self.active:
                self.active[job_id]["cancel_flag"] = True
            self.queue = deque([j for j in self.queue if j["id"] != job_id])

    def get_jobs(self):
        with self.lock:
            return list(self.active.values()) + list(self.queue)


class ModelManager:
    VISION_HINTS = {
        "vl",
        "vision",
        "llava",
        "nanollava",
        "qwen-vl",
        "qwen2-vl",
        "qwen2.5-vl",
        "internvl",
        "jan-vl",
        "pixtral",
        "omni",
        "image",
    }

    STOP_TOKENS = {
        "gguf",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "k",
        "m",
        "s",
        "l",
        "xl",
        "imat",
        "instruct",
        "chat",
        "base",
        "fp16",
        "f16",
        "i",
        "iq",
        "km",
    }

    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = Path(__file__).parent.parent
            self.models_dir = base_dir / "models"
        else:
            self.models_dir = Path(models_dir)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.download_manager = DownloadManager(self.models_dir)
        logger.info("Model directory is set to: %s", self.models_dir.resolve())

    def resolve_model_path(self, filename: str) -> Optional[Path]:
        if not filename:
            return None

        candidate = (self.models_dir / filename).resolve()
        models_root = self.models_dir.resolve()

        try:
            candidate.relative_to(models_root)
        except ValueError:
            return None

        return candidate

    def list_models(self):
        files = [f for f in self.models_dir.iterdir() if f.suffix.lower() == ".gguf"]
        result = []
        for f in files:
            if self._is_mmproj_file(f.name):
                continue
            stat = f.stat()
            vision_info = self.detect_vision_support(f.name)
            result.append(
                {
                    "name": f.name,
                    "size_str": self._human_size(stat.st_size),
                    "quant": self._detect_quant(f.name),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "capabilities": {
                        "vision": vision_info["is_vision"],
                        "mmproj": vision_info["mmproj"],
                        "matcher_score": vision_info["score"],
                        "matcher_reason": vision_info["reason"],
                    },
                }
            )
        return sorted(result, key=lambda x: x["modified"], reverse=True)

    def fetch_hf_repo(self, repo):
        repo = repo.strip()
        if not repo:
            return []

        headers = {"User-Agent": "q1LLM-Local-App"}
        api_url = f"https://huggingface.co/api/models/{repo}"

        try:
            r = requests.get(api_url, headers=headers, timeout=10)
            if r.status_code != 200:
                return []

            data = r.json()
            raw_files = []
            for sib in data.get("siblings", []):
                fname = sib.get("rfilename", "")
                if fname.lower().endswith(".gguf"):
                    raw_files.append(
                        {
                            "name": fname,
                            "url": f"https://huggingface.co/{repo}/resolve/main/{fname}",
                            "quant": self._detect_quant(fname),
                        }
                    )

            def fetch_size(f_obj):
                try:
                    head = requests.head(
                        f_obj["url"], headers=headers, timeout=5, allow_redirects=True
                    )
                    size = int(head.headers.get("Content-Length", 0))
                    f_obj["size_str"] = self._human_size(size)
                except Exception:
                    f_obj["size_str"] = "Unknown"
                return f_obj

            with ThreadPoolExecutor(max_workers=5) as executor:
                files = list(executor.map(fetch_size, raw_files))

            return sorted(files, key=lambda x: x["name"])

        except Exception as e:
            logger.warning("Error fetching HF: %s", e)
            return []

    def delete_model(self, filename):
        try:
            safe_path = self.resolve_model_path(filename)
            if safe_path is None or not safe_path.exists() or not safe_path.is_file():
                return False
            safe_path.unlink()
            return True
        except Exception:
            return False

    def download_async(self, url):
        return self.download_manager.add(url)

    def _human_size(self, size):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    def _detect_quant(self, name):
        match = re.search(r"(Q\d_[A-Z0-9]+|Q\d_K|Q\d_0|F16)", name.upper())
        return match.group(1) if match else "-"

    def _is_mmproj_file(self, filename: str) -> bool:
        return "mmproj" in filename.lower()

    def _normalize_name(self, name: str) -> str:
        stem = Path(name).stem.lower()
        stem = stem.replace("mmproj", "")
        stem = re.sub(r"[-_.]q\d(?:_[a-z0-9]+)?", " ", stem)
        stem = re.sub(r"[-_.]iq\d(?:_[a-z0-9]+)?", " ", stem)
        stem = re.sub(r"[^a-z0-9]+", " ", stem)
        stem = re.sub(r"\s+", " ", stem).strip()
        return stem

    def _name_tokens(self, name: str) -> List[str]:
        normalized = self._normalize_name(name)
        tokens = [t for t in normalized.split(" ") if t and t not in self.STOP_TOKENS and len(t) > 1]
        return tokens

    def _looks_like_vision_model(self, model_filename: str) -> bool:
        lower = model_filename.lower()
        if any(h in lower for h in self.VISION_HINTS):
            return True
        tokens = self._name_tokens(model_filename)
        return any(t in self.VISION_HINTS for t in tokens)

    def _list_mmproj_files(self) -> List[Path]:
        return sorted(
            [f for f in self.models_dir.iterdir() if f.suffix.lower() == ".gguf" and self._is_mmproj_file(f.name)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def _score_mmproj_match(self, model_filename: str, mmproj_filename: str) -> Tuple[int, str]:
        model_norm = self._normalize_name(model_filename)
        mmproj_norm = self._normalize_name(mmproj_filename)
        model_tokens = set(self._name_tokens(model_filename))
        mmproj_tokens = set(self._name_tokens(mmproj_filename))
        common = model_tokens.intersection(mmproj_tokens)

        score = 0
        reason_parts: List[str] = []

        if model_norm and mmproj_norm:
            if model_norm == mmproj_norm:
                score += 100
                reason_parts.append("exact-normalized-name")
            elif model_norm in mmproj_norm or mmproj_norm in model_norm:
                score += 50
                reason_parts.append("normalized-substring")

        if common:
            token_score = min(40, len(common) * 10)
            score += token_score
            reason_parts.append(f"token-overlap:{','.join(sorted(common))}")

        family_hints = ["jan", "qwen", "llava", "internvl", "pixtral", "vision", "vl"]
        for fam in family_hints:
            if fam in model_norm and fam in mmproj_norm:
                score += 20
                reason_parts.append(f"family:{fam}")
                break

        if "high" in model_norm and "high" in mmproj_norm:
            score += 8
            reason_parts.append("tier:high")

        if "base" in model_norm and "base" in mmproj_norm:
            score += 5
            reason_parts.append("tier:base")

        reason = ";".join(reason_parts) if reason_parts else "no-strong-signal"
        return score, reason

    def find_best_mmproj(self, model_filename: str) -> Dict[str, Optional[str]]:
        mmprojs = self._list_mmproj_files()
        if not mmprojs:
            return {"mmproj": None, "score": 0, "reason": "no-mmproj-found"}

        scored: List[Tuple[int, str, Path]] = []
        for mm in mmprojs:
            score, reason = self._score_mmproj_match(model_filename, mm.name)
            scored.append((score, reason, mm))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_reason, best_file = scored[0]

        if best_score >= 20:
            return {
                "mmproj": best_file.name,
                "score": best_score,
                "reason": best_reason,
            }

        if len(mmprojs) == 1 and self._looks_like_vision_model(model_filename):
            return {
                "mmproj": mmprojs[0].name,
                "score": 5,
                "reason": "single-mmproj-fallback-for-vision-model",
            }

        return {"mmproj": None, "score": best_score, "reason": "low-confidence-match"}

    def detect_vision_support(self, model_filename: str) -> Dict[str, Optional[str]]:
        is_vision = self._looks_like_vision_model(model_filename)
        mmproj = self.find_best_mmproj(model_filename)
        mmproj_name = mmproj["mmproj"] if (is_vision or int(mmproj["score"] or 0) >= 80) else None
        return {
            "is_vision": is_vision,
            "mmproj": mmproj_name,
            "score": mmproj["score"],
            "reason": mmproj["reason"],
        }

    def build_load_plan(self, model_filename: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        params = dict(params or {})

        explicit_mmproj = str(params.get("mmproj_path", "") or "").strip()
        disable_auto_mmproj = bool(params.get("disable_auto_mmproj", False))
        chat_format = str(params.get("chat_format", "") or "").strip()

        selected_mmproj = ""
        selection_reason = "disabled"
        selection_score = 0

        if explicit_mmproj:
            selected_mmproj = explicit_mmproj
            selection_reason = "explicit-override"
            selection_score = 1000
        elif not disable_auto_mmproj:
            match = self.find_best_mmproj(model_filename)
            match_score = int(match.get("score") or 0)
            is_vision_like = self._looks_like_vision_model(model_filename)
            if is_vision_like or match_score >= 80:
                selected_mmproj = str(match.get("mmproj") or "")
            else:
                selected_mmproj = ""
            selection_reason = str(match.get("reason") or "auto")
            selection_score = match_score

        mmproj_abs = ""
        if selected_mmproj:
            mmproj_path = self.resolve_model_path(selected_mmproj)
            if mmproj_path is not None and mmproj_path.exists():
                mmproj_abs = str(mmproj_path)

        should_enable_vision = bool(mmproj_abs)
        if should_enable_vision and not chat_format:
            chat_format = "llava-1-5"

        out_params = dict(params)
        out_params.pop("disable_auto_mmproj", None)
        if mmproj_abs:
            out_params["mmproj_path"] = mmproj_abs
        if chat_format:
            out_params["chat_format"] = chat_format

        return {
            "params": out_params,
            "vision_enabled": should_enable_vision,
            "mmproj": Path(mmproj_abs).name if mmproj_abs else "",
            "mmproj_score": selection_score,
            "mmproj_reason": selection_reason,
        }



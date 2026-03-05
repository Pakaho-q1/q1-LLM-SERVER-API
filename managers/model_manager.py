import os
import uuid
import threading
import requests
import hashlib
import re
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


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
            with requests.get(
                job["url"], stream=True, headers=headers, timeout=10
            ) as r:
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
                                    - int(
                                        headers.get("Range", "bytes=0-")
                                        .split("=")[1]
                                        .split("-")[0]
                                    )
                                ) / elapsed
                                remaining = total - downloaded
                                job["eta"] = (
                                    remaining / job["speed"] if job["speed"] > 0 else 0
                                )
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
    def __init__(self, models_dir=None):
        if models_dir is None:

            base_dir = Path(__file__).parent.parent
            self.models_dir = base_dir / "models"
        else:
            self.models_dir = Path(models_dir)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.download_manager = DownloadManager(self.models_dir)
        print(f"📁 Model Directory is set to: {self.models_dir.resolve()}")

    def resolve_model_path(self, filename: str) -> Optional[Path]:
        """Resolve model filename to a safe path under models_dir."""
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
            stat = f.stat()
            result.append(
                {
                    "name": f.name,
                    "size_str": self._human_size(stat.st_size),
                    "quant": self._detect_quant(f.name),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
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
                except:
                    f_obj["size_str"] = "Unknown"
                return f_obj

            with ThreadPoolExecutor(max_workers=5) as executor:
                files = list(executor.map(fetch_size, raw_files))

            return sorted(files, key=lambda x: x["name"])

        except Exception as e:
            print(f"Error fetching HF: {e}")
            return []

    def delete_model(self, filename):
        try:
            safe_path = self.resolve_model_path(filename)
            if safe_path is None or not safe_path.exists() or not safe_path.is_file():
                return False
            safe_path.unlink()
            return True
        except:
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

import gc
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMEngine:
    """
    LLM runtime adapter.

    Backends:
    - llama_server: run llama.cpp server as subprocess and call OpenAI-compatible HTTP APIs
    - python_binding: run llama-cpp-python in-process (legacy fallback)
    """

    def __init__(self):
        self.backend = (
            (os.environ.get("LLM_ENGINE_BACKEND") or "llama_server").strip().lower()
        )
        if self.backend not in {"llama_server", "python_binding"}:
            self.backend = "llama_server"

        self.llm = None
        self.model_name: str = ""
        self.model_path: str = ""
        self.mmproj_path: str = ""
        self.chat_format: str = ""
        self.multimodal_enabled: bool = False
        self.n_ctx: int = 4096
        self.safety_margin: int = 256

        self._server_process: Optional[subprocess.Popen] = None
        self._server_host = os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
        self._server_port = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
        self._server_base_url = f"http://{self._server_host}:{self._server_port}"
        self._server_bin = self._resolve_server_bin()
        self._server_start_timeout = int(
            os.environ.get("LLAMA_SERVER_START_TIMEOUT", "90")
        )

    def _resolve_server_bin(self) -> str:
        explicit = (os.environ.get("LLAMA_SERVER_BIN") or "").strip()
        if explicit:
            return explicit

        server_root = Path(__file__).resolve().parent.parent
        bundled = server_root / "llama_cpp" / "llama-server.exe"
        if bundled.exists():
            return str(bundled)

        return "llama-server"

    def _clamp(self, val, min_v, max_v):
        return max(min_v, min(val, max_v))

    def _validate_gguf_file(self, path: str, label: str) -> Tuple[bool, str]:
        if not os.path.exists(path):
            return False, f"{label} not found: {path}"
        if not os.path.isfile(path):
            return False, f"{label} is not a file: {path}"
        if os.path.getsize(path) < 1024 * 1024:
            return False, f"{label} too small/corrupt: {path}"
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"GGUF":
            return False, f"Invalid GGUF header for {label}: {path}"
        return True, ""

    def _message_has_image(self, message: Dict[str, Any]) -> bool:
        content = message.get("content")
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
        return False

    def _messages_has_image(self, messages: List[Dict[str, Any]]) -> bool:
        return any(self._message_has_image(m) for m in messages)

    def is_loaded(self) -> bool:
        if self.backend == "llama_server":
            return (
                self._server_process is not None
                and self._server_process.poll() is None
                and bool(self.model_name)
            )
        return self.llm is not None

    def _wait_server_ready(self) -> Tuple[bool, str]:
        deadline = time.time() + self._server_start_timeout
        last_err = ""

        while time.time() < deadline:
            if self._server_process is None:
                return False, "llama-server process missing"
            if self._server_process.poll() is not None:
                return (
                    False,
                    f"llama-server exited early with code {self._server_process.returncode}",
                )

            try:
                # Prefer OpenAI-compatible model listing for readiness.
                r = requests.get(f"{self._server_base_url}/v1/models", timeout=2)
                if r.ok:
                    return True, ""
                last_err = f"HTTP {r.status_code}"
            except Exception as exc:
                last_err = str(exc)

            time.sleep(0.5)

        return False, f"Timeout waiting llama-server ready: {last_err}"

    def _start_llama_server(
        self, model_path: str, mmproj_path: str, params: Dict[str, Any]
    ) -> Tuple[bool, str]:
        n_ctx = int(params.get("n_ctx", 4096))
        n_threads = int(params.get("n_threads", 4))
        n_batch = int(params.get("n_batch", 512))
        n_gpu_layers = int(params.get("n_gpu_layers", -1))

        cmd = [
            self._server_bin,
            "-m",
            model_path,
            "--host",
            self._server_host,
            "--port",
            str(self._server_port),
            "-c",
            str(n_ctx),
            "--threads",
            str(n_threads),
            "--batch-size",
            str(n_batch),
        ]

        if n_gpu_layers >= 0:
            cmd.extend(["--n-gpu-layers", str(n_gpu_layers)])

        if mmproj_path:
            cmd.extend(["--mmproj", mmproj_path])

        # Keep startup conservative for production reliability.
        if "flash_attn" in params:
            cmd.extend(
                ["--flash-attn", "on" if bool(params.get("flash_attn")) else "off"]
            )

        try:
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return False, (
                f"llama-server binary not found: '{self._server_bin}'. "
                "Set LLAMA_SERVER_BIN to absolute path of llama-server executable."
            )
        except Exception as exc:
            return False, f"Failed to start llama-server: {exc}"

        ok, reason = self._wait_server_ready()
        if not ok:
            self._stop_llama_server()
            return False, reason

        return True, ""

    def _stop_llama_server(self) -> None:
        proc = self._server_process
        self._server_process = None
        if proc is None:
            return

        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

    def _load_python_binding(
        self,
        model_path: str,
        mmproj_path: str,
        chat_format: str,
        params: Dict[str, Any],
    ) -> Tuple[bool, str]:
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import (
                Llava15ChatHandler,
                MoondreamChatHandler,
                NanoLlavaChatHandler,
            )
        except Exception as exc:
            return False, f"python_binding backend unavailable: {exc}"

        try:
            seed = int(params.get("seed", -1))
        except Exception:
            seed = -1

        llm_kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": int(params.get("n_ctx", 4096)),
            "n_gpu_layers": int(params.get("n_gpu_layers", 0)),
            "n_threads": int(params.get("n_threads", 4)),
            "n_batch": int(params.get("n_batch", 512)),
            "seed": seed,
            "verbose": False,
        }

        if mmproj_path:
            hint = (chat_format or "").strip().lower()
            if hint in {"nanollava", "nano_llava"}:
                llm_kwargs["chat_handler"] = NanoLlavaChatHandler(
                    clip_model_path=mmproj_path
                )
            elif hint in {"moondream"}:
                llm_kwargs["chat_handler"] = MoondreamChatHandler(
                    clip_model_path=mmproj_path
                )
            else:
                llm_kwargs["chat_handler"] = Llava15ChatHandler(
                    clip_model_path=mmproj_path
                )

        for bool_key, default in [
            ("f16_kv", True),
            ("use_mmap", True),
            ("use_mlock", False),
            ("flash_attn", True),
        ]:
            llm_kwargs[bool_key] = bool(params.get(bool_key, default))

        if "rope_scaling_type" in params:
            llm_kwargs["rope_scaling_type"] = params["rope_scaling_type"]
        if "rope_freq_base" in params:
            llm_kwargs["rope_freq_base"] = float(params["rope_freq_base"])
        if "rope_freq_scale" in params:
            llm_kwargs["rope_freq_scale"] = float(params["rope_freq_scale"])

        self.llm = Llama(**llm_kwargs)
        return True, ""

    def load_model(self, model_path: str, **params) -> Tuple[bool, str]:
        try:
            self.unload_model()

            ok, err = self._validate_gguf_file(model_path, "Model file")
            if not ok:
                return False, err

            mmproj_path = str(params.get("mmproj_path", "") or "").strip()
            chat_format = str(params.get("chat_format", "") or "").strip()
            if mmproj_path:
                ok, err = self._validate_gguf_file(mmproj_path, "mmproj file")
                if not ok:
                    return False, err

            self.n_ctx = int(params.get("n_ctx", 4096))

            if self.backend == "llama_server":
                success, reason = self._start_llama_server(
                    model_path, mmproj_path, params
                )
                if not success:
                    return False, reason
            else:
                success, reason = self._load_python_binding(
                    model_path, mmproj_path, chat_format, params
                )
                if not success:
                    return False, reason

            self.model_path = model_path
            self.model_name = os.path.basename(model_path)
            self.mmproj_path = mmproj_path
            self.chat_format = chat_format
            self.multimodal_enabled = bool(mmproj_path)

            mode = (
                "llama_server" if self.backend == "llama_server" else "python_binding"
            )
            if self.multimodal_enabled:
                return True, f"Loaded multimodal model ({mode}): {self.model_name}"
            return True, f"Loaded model ({mode}): {self.model_name}"

        except Exception as e:
            logger.error("Load Error: %s", e, exc_info=True)
            return False, str(e)

    def unload_model(self):
        if self.backend == "llama_server":
            self._stop_llama_server()
        else:
            if self.llm:
                del self.llm
                self.llm = None
                gc.collect()
                logger.info("Model unloaded.")

        self.model_name = ""
        self.model_path = ""
        self.mmproj_path = ""
        self.chat_format = ""
        self.multimodal_enabled = False

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0

        if self.backend == "llama_server" and self.is_loaded():
            try:
                r = requests.post(
                    f"{self._server_base_url}/tokenize",
                    json={"content": text},
                    timeout=5,
                )
                if r.ok:
                    data = r.json()
                    if isinstance(data, dict):
                        if isinstance(data.get("tokens"), list):
                            return len(data["tokens"])
                        if isinstance(data.get("n_tokens"), int):
                            return int(data["n_tokens"])
            except Exception:
                pass

        if self.llm is not None:
            try:
                return len(self.llm.tokenize(text.encode("utf-8")))
            except Exception as e:
                logger.warning("Tokenize failed, using estimation. Error: %s", e)

        return max(1, len(text) // 3)

    def generate_json(
        self, messages: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> str:
        if not self.is_loaded():
            return "{}"

        max_tokens = int(params.get("max_tokens", 1024))

        if self.backend == "llama_server":
            payload = {
                "model": self.model_name or "local-model",
                "messages": messages,
                "stream": False,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            try:
                r = requests.post(
                    f"{self._server_base_url}/v1/chat/completions",
                    json=payload,
                    timeout=120,
                )
                if not r.ok:
                    return "{}"
                data = json.loads(r.content.decode("utf-8", errors="replace"))
                return str(
                    data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                )
            except Exception:
                return "{}"

        call_kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

        try:
            self.llm.reset()
            response = self.llm.create_chat_completion(**call_kwargs)
            content = response["choices"][0]["message"]["content"]
            logger.info("[mem0 Extraction]: %s", content)
            return content
        except Exception as e:
            logger.error("JSON Generation Error: %s", e, exc_info=True)
            return "{}"

    def _stream_from_llama_server(
        self, messages: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> Generator[str, None, None]:
        payload: Dict[str, Any] = {
            "model": self.model_name or params.get("model") or "local-model",
            "messages": messages,
            "stream": True,
            "max_tokens": int(params.get("max_tokens", 512)),
            "temperature": self._clamp(float(params.get("temperature", 0.7)), 0.0, 2.0),
            "top_p": self._clamp(float(params.get("top_p", 0.95)), 0.0, 1.0),
            "stop": params.get("stop"),
        }
        if params.get("seed") is not None:
            try:
                payload["seed"] = int(params.get("seed"))
            except Exception:
                pass

        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            with requests.post(
                f"{self._server_base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                if resp.status_code >= 400:
                    msg = resp.text[:500]
                    yield f"\n[Backend Error: llama-server HTTP {resp.status_code}: {msg}]"
                    return

                for raw in resp.iter_lines(decode_unicode=False):
                    if not raw:
                        continue
                    if isinstance(raw, bytes):
                        line = raw.decode("utf-8", errors="replace").strip()
                    else:
                        line = str(raw).strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if not body or body == "[DONE]":
                        if body == "[DONE]":
                            break
                        continue
                    try:
                        obj = json.loads(body)
                    except Exception:
                        continue

                    if isinstance(obj, dict) and obj.get("error"):
                        msg = obj.get("error", {}).get("message") or str(
                            obj.get("error")
                        )
                        yield f"\n[Backend Error: {msg}]"
                        return

                    chunk = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if chunk:
                        yield str(chunk)
        except Exception as e:
            logger.error("Generation Error (llama_server): %s", e, exc_info=True)
            yield f"\n[Backend Error: {str(e)}]"

    def completion_stream(
        self,
        messages: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Generator[str, None, None]:
        if not self.is_loaded():
            yield "No model loaded."
            return

        if self._messages_has_image(messages) and not self.multimodal_enabled:
            yield "\n[Backend Error: Image input requires a vision model with mmproj loaded]"
            return

        if self.backend == "llama_server":
            yield from self._stream_from_llama_server(messages, params)
            return

        max_tokens = int(params.get("max_tokens", 512))
        temperature = self._clamp(float(params.get("temperature", 0.7)), 0.0, 2.0)
        top_k = int(params.get("top_k", 40))
        top_p = self._clamp(float(params.get("top_p", 0.95)), 0.0, 1.0)
        min_p = self._clamp(float(params.get("min_p", 0.0)), 0.0, 1.0)
        typical_p = self._clamp(float(params.get("typical_p", 1.0)), 0.0, 1.0)
        tfs_z = float(params.get("tfs_z", 1.0))
        repeat_penalty = float(params.get("repeat_penalty", 1.1))
        freq_penalty = float(params.get("frequency_penalty", 0.0))
        pres_penalty = float(params.get("presence_penalty", 0.0))
        mirostat_mode = int(params.get("mirostat_mode", 0))
        mirostat_tau = float(params.get("mirostat_tau", 5.0))
        mirostat_eta = float(params.get("mirostat_eta", 0.1))

        call_kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "typical_p": typical_p,
            "tfs_z": tfs_z,
            "repeat_penalty": repeat_penalty,
            "frequency_penalty": freq_penalty,
            "presence_penalty": pres_penalty,
            "mirostat_mode": mirostat_mode,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            "logit_bias": params.get("logit_bias"),
            "grammar": params.get("grammar"),
            "stop": params.get("stop"),
            "stream": True,
        }

        if params.get("seed") is not None:
            try:
                call_kwargs["seed"] = int(params["seed"])
            except Exception:
                pass

        call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}
        call_kwargs.pop("penalty_last_n", None)

        try:
            stream = self.llm.create_chat_completion(**call_kwargs)
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        except Exception as e:
            logger.error("Generation Error: %s", e, exc_info=True)
            yield f"\n[Backend Error: {str(e)}]"

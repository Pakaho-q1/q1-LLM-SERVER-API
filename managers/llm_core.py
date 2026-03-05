import os
import gc
import logging
from typing import List, Dict, Generator, Optional, Any, Tuple
from llama_cpp import Llama

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Production-Ready Llama Engine
    - Full Sampling Stack
    - Context Sliding Window
    - Deterministic Mode
    - Hardware Tuning
    """

    def __init__(self):
        self.llm: Optional[Llama] = None
        self.model_name: str = ""
        self.model_path: str = ""
        self.n_ctx: int = 4096
        self.safety_margin: int = 256

    def _clamp(self, val, min_v, max_v):
        return max(min_v, min(val, max_v))

    def load_model(self, model_path: str, **params) -> Tuple[bool, str]:
        try:
            self.unload_model()

            if not os.path.exists(model_path):
                return False, f"❌ File not found: {model_path}"

            self.n_ctx = int(params.get("n_ctx", 4096))

            try:
                seed = int(params.get("seed", -1))
            except Exception:
                seed = -1

            llm_kwargs = {
                "model_path": model_path,
                "n_ctx": self.n_ctx,
                "n_gpu_layers": int(params.get("n_gpu_layers", -1)),
                "n_threads": int(params.get("n_threads", 4)),
                "n_batch": int(params.get("n_batch", 512)),
                "seed": seed,
                "verbose": False,
            }

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

            self.model_path = model_path
            self.model_name = os.path.basename(model_path)

            return True, f"✅ Loaded: {self.model_name}"

        except Exception as e:
            logger.error(f"Load Error: {e}", exc_info=True)
            return False, str(e)

    def unload_model(self):
        if self.llm:
            del self.llm
            self.llm = None
            gc.collect()
            logger.info("Model unloaded.")
        self.model_name = ""
        self.model_path = ""

    def count_tokens(self, text: str) -> int:
        if not self.llm:
            return 0
        try:
            return len(self.llm.tokenize(text.encode("utf-8")))
        except Exception as e:
            logger.warning(f"Tokenize failed, using estimation. Error: {e}")
            return len(text) // 3

    def generate_json(
        self, messages: List[Dict[str, str]], params: Dict[str, Any]
    ) -> str:
        if not self.llm:
            return "{}"

        call_kwargs = {
            "messages": messages,
            "max_tokens": int(params.get("max_tokens", 1024)),
            "temperature": 0.0,
            "stream": False,
            "response_format": {"type": "json_object"},
        }

        try:
            self.llm.reset()
            response = self.llm.create_chat_completion(**call_kwargs)
            content = response["choices"][0]["message"]["content"]
            logger.info(f"\n[mem0 Extraction]: {content}\n")
            return content
        except Exception as e:
            logger.error(f"JSON Generation Error: {e}", exc_info=True)
            return "{}"

    def completion_stream(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> Generator[str, None, None]:

        if not self.llm:
            yield "⚠️ No model loaded."
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
        for unsupported_key in ["penalty_last_n"]:
            call_kwargs.pop(unsupported_key, None)

        try:
            stream = self.llm.create_chat_completion(**call_kwargs)
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        except Exception as e:
            logger.error(f"Generation Error: {e}", exc_info=True)
            yield f"\n[Backend Error: {str(e)}]"

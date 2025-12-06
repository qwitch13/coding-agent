"""
Local AI Support for Coding Agent
Optimized for Apple Silicon M3/M4 chips using MLX, llama.cpp, or Ollama.
"""
import asyncio
import json
import os
import platform
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
import aiohttp

from config import config, LocalAIConfig, MODELS_DIR, CACHE_DIR


@dataclass
class LocalModelInfo:
    """Information about a local model."""
    name: str
    path: str
    size_gb: float
    backend: str
    quantization: str = "Q4_K_M"
    context_size: int = 4096


# Popular coding models optimized for local inference
RECOMMENDED_MODELS = {
    "codellama-7b": LocalModelInfo(
        name="CodeLlama 7B",
        path="codellama-7b-instruct.Q4_K_M.gguf",
        size_gb=4.1,
        backend="mlx",
        quantization="Q4_K_M",
        context_size=4096
    ),
    "codellama-13b": LocalModelInfo(
        name="CodeLlama 13B",
        path="codellama-13b-instruct.Q4_K_M.gguf",
        size_gb=7.9,
        backend="mlx",
        quantization="Q4_K_M",
        context_size=4096
    ),
    "deepseek-coder-6.7b": LocalModelInfo(
        name="DeepSeek Coder 6.7B",
        path="deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        size_gb=4.0,
        backend="mlx",
        quantization="Q4_K_M",
        context_size=8192
    ),
    "qwen2.5-coder-7b": LocalModelInfo(
        name="Qwen2.5 Coder 7B",
        path="qwen2.5-coder-7b-instruct.Q4_K_M.gguf",
        size_gb=4.5,
        backend="mlx",
        quantization="Q4_K_M",
        context_size=8192
    ),
    "starcoder2-7b": LocalModelInfo(
        name="StarCoder2 7B",
        path="starcoder2-7b-instruct.Q4_K_M.gguf",
        size_gb=4.2,
        backend="mlx",
        quantization="Q4_K_M",
        context_size=4096
    ),
}


class BaseLocalBackend(ABC):
    """Abstract base class for local AI backends."""

    def __init__(self, config: LocalAIConfig):
        self.config = config
        self.model_loaded = False

    @abstractmethod
    async def load_model(self, model_path: str) -> bool:
        """Load a model into memory."""
        pass

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.7, stream: bool = False) -> str:
        """Generate text from the model."""
        pass

    @abstractmethod
    async def unload_model(self):
        """Unload the current model from memory."""
        pass

    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        pass


class MLXBackend(BaseLocalBackend):
    """
    MLX Backend for Apple Silicon.
    MLX is Apple's machine learning framework optimized for M1/M2/M3/M4 chips.
    """

    def __init__(self, local_config: Optional[LocalAIConfig] = None):
        cfg = local_config or config.local_ai
        super().__init__(cfg)
        self.model = None
        self.tokenizer = None
        self._mlx_available = self._check_mlx()

    def _check_mlx(self) -> bool:
        """Check if MLX is available."""
        try:
            import mlx.core as mx
            import mlx_lm
            return True
        except ImportError:
            return False

    async def load_model(self, model_path: str) -> bool:
        """Load model using MLX."""
        if not self._mlx_available:
            raise RuntimeError("MLX not available. Install with: pip install mlx mlx-lm")

        try:
            from mlx_lm import load

            # Check if model path is a HuggingFace model ID or local path
            if os.path.exists(model_path):
                self.model, self.tokenizer = load(model_path)
            else:
                # Assume it's a HuggingFace model ID
                self.model, self.tokenizer = load(model_path)

            self.model_loaded = True
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to load model with MLX: {e}")

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.7, stream: bool = False) -> str:
        """Generate text using MLX."""
        if not self.model_loaded:
            raise RuntimeError("No model loaded")

        try:
            from mlx_lm import generate

            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )

            return response

        except Exception as e:
            raise RuntimeError(f"MLX generation failed: {e}")

    async def unload_model(self):
        """Unload model from memory."""
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        # Force garbage collection
        import gc
        gc.collect()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get MLX memory usage."""
        try:
            import mlx.core as mx
            # MLX doesn't have direct memory tracking, estimate from model size
            return {
                "estimated_gb": 4.0 if self.model_loaded else 0.0,
                "backend": "mlx"
            }
        except:
            return {"estimated_gb": 0.0, "backend": "mlx"}


class LlamaCppBackend(BaseLocalBackend):
    """
    llama.cpp Backend for efficient CPU/GPU inference.
    Works on all platforms including Apple Silicon.
    """

    def __init__(self, local_config: Optional[LocalAIConfig] = None):
        cfg = local_config or config.local_ai
        super().__init__(cfg)
        self.llm = None
        self._llama_cpp_available = self._check_llama_cpp()

    def _check_llama_cpp(self) -> bool:
        """Check if llama-cpp-python is available."""
        try:
            from llama_cpp import Llama
            return True
        except ImportError:
            return False

    async def load_model(self, model_path: str) -> bool:
        """Load model using llama.cpp."""
        if not self._llama_cpp_available:
            raise RuntimeError("llama-cpp-python not available. Install with: pip install llama-cpp-python")

        try:
            from llama_cpp import Llama

            # Determine GPU layers based on config
            n_gpu_layers = self.config.gpu_layers
            if n_gpu_layers == -1:
                # Auto-detect: use all layers on Apple Silicon
                if platform.processor() == 'arm' and platform.system() == 'Darwin':
                    n_gpu_layers = 99  # Offload all layers to Metal

            self.llm = Llama(
                model_path=model_path,
                n_ctx=self.config.context_size,
                n_threads=self.config.threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )

            self.model_loaded = True
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to load model with llama.cpp: {e}")

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.7, stream: bool = False) -> str:
        """Generate text using llama.cpp."""
        if not self.model_loaded or not self.llm:
            raise RuntimeError("No model loaded")

        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[/INST]", "```\n\n"],
                echo=False
            )

            return response["choices"][0]["text"]

        except Exception as e:
            raise RuntimeError(f"llama.cpp generation failed: {e}")

    async def unload_model(self):
        """Unload model from memory."""
        self.llm = None
        self.model_loaded = False

        import gc
        gc.collect()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get llama.cpp memory usage."""
        if not self.model_loaded:
            return {"estimated_gb": 0.0, "backend": "llama.cpp"}

        # Estimate based on context size
        ctx_mem = (self.config.context_size * 4 * 32) / (1024 ** 3)  # Rough estimate
        return {
            "estimated_gb": 4.0 + ctx_mem,
            "backend": "llama.cpp"
        }


class OllamaBackend(BaseLocalBackend):
    """
    Ollama Backend for easy local model management.
    Ollama handles model downloading and serving automatically.
    """

    def __init__(self, local_config: Optional[LocalAIConfig] = None):
        cfg = local_config or config.local_ai
        super().__init__(cfg)
        self.base_url = "http://localhost:11434"
        self.current_model = None

    async def _check_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except:
            return False

    async def _start_ollama(self) -> bool:
        """Attempt to start Ollama server."""
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            await asyncio.sleep(2)  # Wait for server to start
            return await self._check_ollama_running()
        except:
            return False

    async def load_model(self, model_name: str) -> bool:
        """Load/pull model using Ollama."""
        if not await self._check_ollama_running():
            if not await self._start_ollama():
                raise RuntimeError("Ollama not running. Start with: ollama serve")

        try:
            # Pull model if not available
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 min for download
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Failed to pull model: {await response.text()}")

            self.current_model = model_name
            self.model_loaded = True
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to load model with Ollama: {e}")

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.7, stream: bool = False) -> str:
        """Generate text using Ollama."""
        if not self.model_loaded:
            raise RuntimeError("No model loaded")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.current_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    data = await response.json()
                    return data.get("response", "")

        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    async def unload_model(self):
        """Unload model (Ollama manages this automatically)."""
        self.current_model = None
        self.model_loaded = False

    def get_memory_usage(self) -> Dict[str, float]:
        """Get Ollama memory usage."""
        return {
            "estimated_gb": 4.0 if self.model_loaded else 0.0,
            "backend": "ollama",
            "model": self.current_model
        }


class LocalAIManager:
    """
    Manages local AI inference on Apple Silicon and other platforms.
    Automatically selects the best backend for the current hardware.
    """

    def __init__(self):
        self.backend: Optional[BaseLocalBackend] = None
        self.current_model: Optional[str] = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the best available backend."""
        local_config = config.local_ai

        if not local_config.enabled:
            return

        backend_name = local_config.backend.lower()

        if backend_name == "mlx":
            self.backend = MLXBackend(local_config)
        elif backend_name == "llama.cpp" or backend_name == "llamacpp":
            self.backend = LlamaCppBackend(local_config)
        elif backend_name == "ollama":
            self.backend = OllamaBackend(local_config)
        else:
            # Auto-detect best backend
            if config.is_apple_silicon():
                # Prefer MLX on Apple Silicon
                try:
                    self.backend = MLXBackend(local_config)
                except:
                    self.backend = OllamaBackend(local_config)
            else:
                self.backend = OllamaBackend(local_config)

    async def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load a local model.

        Args:
            model_name: Model name/path. If None, uses config default.

        Returns:
            True if loaded successfully
        """
        if not self.backend:
            raise RuntimeError("No backend available")

        model = model_name or config.local_ai.model_name

        # Check if it's a recommended model
        if model in RECOMMENDED_MODELS:
            model_info = RECOMMENDED_MODELS[model]
            model_path = MODELS_DIR / model_info.path

            # Check if model exists locally
            if not model_path.exists():
                raise RuntimeError(
                    f"Model not found: {model_path}\n"
                    f"Download from HuggingFace or use Ollama: ollama pull {model}"
                )

            model = str(model_path)

        success = await self.backend.load_model(model)
        if success:
            self.current_model = model

        return success

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.7) -> str:
        """Generate text using the local model."""
        if not self.backend or not self.backend.model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")

        return await self.backend.generate(prompt, max_tokens, temperature)

    async def generate_code_fix(self, code: str, error: str,
                                 context: Optional[str] = None) -> str:
        """Generate a code fix using the local model."""
        prompt = f"""<|system|>
You are an expert software engineer. Fix the code error. Return only the fixed code.
<|end|>
<|user|>
Fix this code:

ERROR: {error}

CODE:
```
{code}
```
{f'CONTEXT: {context}' if context else ''}
<|end|>
<|assistant|>
"""
        return await self.generate(prompt)

    async def analyze_code(self, code: str, analysis_type: str = "review") -> str:
        """Analyze code using the local model."""
        prompt = f"""<|system|>
You are an expert code reviewer. Analyze the code and provide actionable feedback.
<|end|>
<|user|>
Analyze this code for {analysis_type}:

```
{code}
```

Provide:
1. Issues found
2. Suggestions for improvement
3. Optimized code if applicable
<|end|>
<|assistant|>
"""
        return await self.generate(prompt)

    async def unload(self):
        """Unload the current model."""
        if self.backend:
            await self.backend.unload_model()
            self.current_model = None

    def get_status(self) -> Dict[str, Any]:
        """Get current status of local AI."""
        return {
            "enabled": config.local_ai.enabled,
            "backend": config.local_ai.backend,
            "model_loaded": self.backend.model_loaded if self.backend else False,
            "current_model": self.current_model,
            "memory_usage": self.backend.get_memory_usage() if self.backend else {},
            "is_apple_silicon": config.is_apple_silicon()
        }

    @staticmethod
    def get_recommended_models() -> Dict[str, LocalModelInfo]:
        """Get list of recommended models for coding."""
        return RECOMMENDED_MODELS

    @staticmethod
    async def download_model(model_name: str, backend: str = "ollama") -> bool:
        """
        Download a model for local use.

        Args:
            model_name: Name of the model to download
            backend: Backend to use for downloading (ollama recommended)

        Returns:
            True if download successful
        """
        if backend == "ollama":
            try:
                process = await asyncio.create_subprocess_exec(
                    "ollama", "pull", model_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                return process.returncode == 0
            except:
                return False

        return False


# Global local AI manager instance
local_ai_manager = LocalAIManager()

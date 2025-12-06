"""
Configuration for Coding Agent
Manages all settings, API keys, and provider configurations.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json

# Base directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = BASE_DIR / "logs"
JOURNAL_DIR = BASE_DIR / "journal"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / ".cache"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for d in [LOGS_DIR, JOURNAL_DIR, RESULTS_DIR, CACHE_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

# Config file path
CONFIG_FILE = BASE_DIR / "agent_config.json"


@dataclass
class AIProviderConfig:
    """Configuration for an AI provider."""
    name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = ""
    enabled: bool = True
    max_tokens: int = 4096
    temperature: float = 0.7


@dataclass
class LocalAIConfig:
    """Configuration for local AI models (M3/M4 Apple Silicon)."""
    enabled: bool = True
    backend: str = "mlx"  # mlx, llama.cpp, ollama
    model_path: Optional[str] = None
    model_name: str = "codellama-7b"
    context_size: int = 4096
    gpu_layers: int = -1  # -1 = all layers on GPU
    threads: int = 8


@dataclass
class GitConfig:
    """Git and GitHub configuration."""
    auto_commit: bool = False
    auto_push: bool = False
    default_branch: str = "main"
    commit_message_template: str = "[Agent] {action}: {description}"
    github_accounts: List[str] = field(default_factory=lambda: ["nebulai13", "qwitch13"])


@dataclass
class AgentConfig:
    """Main agent configuration."""
    # Operation mode
    mode: str = "standalone"  # standalone, single_ai, multi_agent

    # Optimization loop settings
    max_iterations: int = 10
    test_command: str = "pytest"
    build_command: str = "python -m py_compile"
    lint_command: str = "ruff check"

    # Parallelism
    max_parallel_agents: int = 4

    # Timeouts (seconds)
    test_timeout: int = 300
    build_timeout: int = 120
    ai_timeout: int = 60

    # Journaling
    journal_enabled: bool = True
    verbose: bool = False


class Config:
    """Central configuration manager."""

    # Default AI provider configurations
    PROVIDERS = {
        "claude": AIProviderConfig(
            name="Claude",
            api_base="https://api.anthropic.com/v1",
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
        ),
        "chatgpt": AIProviderConfig(
            name="ChatGPT",
            api_base="https://api.openai.com/v1",
            model="gpt-4o",
            max_tokens=8192,
        ),
        "gemini": AIProviderConfig(
            name="Gemini",
            api_base="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.0-flash",
            max_tokens=8192,
        ),
        "perplexity": AIProviderConfig(
            name="Perplexity",
            api_base="https://api.perplexity.ai",
            model="llama-3.1-sonar-large-128k-online",
            max_tokens=4096,
        ),
    }

    # Environment variable names for API keys
    API_KEY_ENV_VARS = {
        "claude": "ANTHROPIC_API_KEY",
        "chatgpt": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
    }

    def __init__(self):
        self.agent = AgentConfig()
        self.local_ai = LocalAIConfig()
        self.git = GitConfig()
        self.providers = dict(self.PROVIDERS)
        self._load_config()
        self._load_api_keys()

    def _load_config(self):
        """Load configuration from file if exists."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)

                # Update agent config
                if "agent" in data:
                    for key, value in data["agent"].items():
                        if hasattr(self.agent, key):
                            setattr(self.agent, key, value)

                # Update local AI config
                if "local_ai" in data:
                    for key, value in data["local_ai"].items():
                        if hasattr(self.local_ai, key):
                            setattr(self.local_ai, key, value)

                # Update git config
                if "git" in data:
                    for key, value in data["git"].items():
                        if hasattr(self.git, key):
                            setattr(self.git, key, value)

            except Exception as e:
                print(f"Warning: Could not load config file: {e}")

    def _load_api_keys(self):
        """Load API keys from environment variables."""
        for provider_id, env_var in self.API_KEY_ENV_VARS.items():
            if provider_id in self.providers:
                api_key = os.environ.get(env_var)
                if api_key:
                    self.providers[provider_id].api_key = api_key

    def save_config(self):
        """Save current configuration to file."""
        data = {
            "agent": {
                "mode": self.agent.mode,
                "max_iterations": self.agent.max_iterations,
                "test_command": self.agent.test_command,
                "build_command": self.agent.build_command,
                "lint_command": self.agent.lint_command,
                "max_parallel_agents": self.agent.max_parallel_agents,
                "test_timeout": self.agent.test_timeout,
                "build_timeout": self.agent.build_timeout,
                "ai_timeout": self.agent.ai_timeout,
                "journal_enabled": self.agent.journal_enabled,
                "verbose": self.agent.verbose,
            },
            "local_ai": {
                "enabled": self.local_ai.enabled,
                "backend": self.local_ai.backend,
                "model_path": self.local_ai.model_path,
                "model_name": self.local_ai.model_name,
                "context_size": self.local_ai.context_size,
                "gpu_layers": self.local_ai.gpu_layers,
                "threads": self.local_ai.threads,
            },
            "git": {
                "auto_commit": self.git.auto_commit,
                "auto_push": self.git.auto_push,
                "default_branch": self.git.default_branch,
                "commit_message_template": self.git.commit_message_template,
                "github_accounts": self.git.github_accounts,
            }
        }

        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def get_provider(self, name: str) -> Optional[AIProviderConfig]:
        """Get a provider configuration by name."""
        return self.providers.get(name.lower())

    def get_enabled_providers(self) -> Dict[str, AIProviderConfig]:
        """Get all enabled providers with API keys."""
        return {
            name: config for name, config in self.providers.items()
            if config.enabled and config.api_key
        }

    def is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon (M1/M2/M3/M4)."""
        import platform
        return platform.processor() == 'arm' and platform.system() == 'Darwin'

    def get_optimal_local_config(self) -> LocalAIConfig:
        """Get optimal local AI configuration for current hardware."""
        import platform

        config = LocalAIConfig()

        if self.is_apple_silicon():
            # Apple Silicon optimizations
            config.enabled = True
            config.backend = "mlx"  # MLX is optimized for Apple Silicon
            config.gpu_layers = -1  # Use all GPU layers

            # Detect chip variant for optimal settings
            # M3/M4 have more neural engine cores
            chip_info = platform.processor()
            config.threads = os.cpu_count() or 8

        else:
            # Fallback for non-Apple Silicon
            config.backend = "llama.cpp"
            config.gpu_layers = 0

        return config


# Global config instance
config = Config()

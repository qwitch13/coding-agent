"""
AI Provider Integrations for Coding Agent
Supports Claude, ChatGPT, Gemini, Perplexity, and Junie.
"""
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Generator
import asyncio
import aiohttp

from config import config, AIProviderConfig


@dataclass
class AIMessage:
    """Represents a message in the conversation."""
    role: str  # system, user, assistant
    content: str


@dataclass
class AIResponse:
    """Represents a response from an AI provider."""
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0
    raw_response: Optional[Dict] = None


class BaseAIProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, config: AIProviderConfig):
        self.config = config
        self.name = config.name
        self.conversation_history: List[AIMessage] = []

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       stream: bool = False) -> AIResponse:
        """Generate a response from the AI."""
        pass

    @abstractmethod
    async def generate_code_fix(self, code: str, error: str,
                                 context: Optional[str] = None) -> AIResponse:
        """Generate a code fix for the given error."""
        pass

    @abstractmethod
    async def analyze_code(self, code: str, analysis_type: str = "review") -> AIResponse:
        """Analyze code for issues, optimizations, etc."""
        pass

    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append(AIMessage(role=role, content=content))

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_history_as_messages(self) -> List[Dict[str, str]]:
        """Get conversation history as list of message dicts."""
        return [{"role": msg.role, "content": msg.content}
                for msg in self.conversation_history]


class ClaudeProvider(BaseAIProvider):
    """Anthropic Claude API provider."""

    def __init__(self, provider_config: Optional[AIProviderConfig] = None):
        cfg = provider_config or config.get_provider("claude")
        super().__init__(cfg)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       stream: bool = False) -> AIResponse:
        """Generate using Claude API."""
        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01"
        }

        messages = self.get_history_as_messages()
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.api_base}/messages",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()

        latency = (time.time() - start_time) * 1000

        if "content" in data and len(data["content"]) > 0:
            content = data["content"][0].get("text", "")
            self.add_to_history("user", prompt)
            self.add_to_history("assistant", content)

            return AIResponse(
                content=content,
                provider="claude",
                model=self.config.model,
                tokens_used=data.get("usage", {}).get("output_tokens", 0),
                latency_ms=latency,
                raw_response=data
            )

        raise Exception(f"Claude API error: {data}")

    async def generate_code_fix(self, code: str, error: str,
                                 context: Optional[str] = None) -> AIResponse:
        """Generate a code fix using Claude."""
        system_prompt = """You are an expert software engineer. Your task is to fix code errors.
Analyze the error, understand the root cause, and provide a corrected version of the code.
Return ONLY the fixed code without explanations unless specifically asked."""

        prompt = f"""Fix the following code that has this error:

ERROR:
{error}

CODE:
```
{code}
```
"""
        if context:
            prompt += f"\nADDITIONAL CONTEXT:\n{context}"

        return await self.generate(prompt, system_prompt)

    async def analyze_code(self, code: str, analysis_type: str = "review") -> AIResponse:
        """Analyze code using Claude."""
        prompts = {
            "review": "Review this code for bugs, issues, and potential improvements:",
            "optimize": "Analyze this code for performance optimizations:",
            "security": "Analyze this code for security vulnerabilities:",
            "refactor": "Suggest refactoring improvements for this code:",
        }

        system_prompt = """You are an expert code reviewer. Provide detailed, actionable feedback.
Format your response as:
1. ISSUES: List any bugs or problems found
2. SUGGESTIONS: List improvements
3. OPTIMIZED_CODE: If applicable, provide improved code"""

        prompt = f"""{prompts.get(analysis_type, prompts['review'])}

```
{code}
```"""

        return await self.generate(prompt, system_prompt)


class ChatGPTProvider(BaseAIProvider):
    """OpenAI ChatGPT API provider."""

    def __init__(self, provider_config: Optional[AIProviderConfig] = None):
        cfg = provider_config or config.get_provider("chatgpt")
        super().__init__(cfg)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       stream: bool = False) -> AIResponse:
        """Generate using ChatGPT API."""
        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.get_history_as_messages())
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()

        latency = (time.time() - start_time) * 1000

        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            self.add_to_history("user", prompt)
            self.add_to_history("assistant", content)

            return AIResponse(
                content=content,
                provider="chatgpt",
                model=self.config.model,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                latency_ms=latency,
                raw_response=data
            )

        raise Exception(f"ChatGPT API error: {data}")

    async def generate_code_fix(self, code: str, error: str,
                                 context: Optional[str] = None) -> AIResponse:
        """Generate a code fix using ChatGPT."""
        system_prompt = """You are an expert software engineer. Your task is to fix code errors.
Analyze the error, understand the root cause, and provide a corrected version of the code.
Return ONLY the fixed code without explanations unless specifically asked."""

        prompt = f"""Fix the following code that has this error:

ERROR:
{error}

CODE:
```
{code}
```
"""
        if context:
            prompt += f"\nADDITIONAL CONTEXT:\n{context}"

        return await self.generate(prompt, system_prompt)

    async def analyze_code(self, code: str, analysis_type: str = "review") -> AIResponse:
        """Analyze code using ChatGPT."""
        prompts = {
            "review": "Review this code for bugs, issues, and potential improvements:",
            "optimize": "Analyze this code for performance optimizations:",
            "security": "Analyze this code for security vulnerabilities:",
            "refactor": "Suggest refactoring improvements for this code:",
        }

        system_prompt = """You are an expert code reviewer. Provide detailed, actionable feedback."""

        prompt = f"""{prompts.get(analysis_type, prompts['review'])}

```
{code}
```"""

        return await self.generate(prompt, system_prompt)


class GeminiProvider(BaseAIProvider):
    """Google Gemini API provider."""

    def __init__(self, provider_config: Optional[AIProviderConfig] = None):
        cfg = provider_config or config.get_provider("gemini")
        super().__init__(cfg)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       stream: bool = False) -> AIResponse:
        """Generate using Gemini API."""
        start_time = time.time()

        # Build contents for Gemini API
        contents = []

        # Add system instruction if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Add conversation history
        for msg in self.conversation_history:
            role = "user" if msg.role == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": full_prompt}]
        })

        payload = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
        }

        url = f"{self.config.api_base}/models/{self.config.model}:generateContent?key={self.config.api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()

        latency = (time.time() - start_time) * 1000

        if "candidates" in data and len(data["candidates"]) > 0:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            self.add_to_history("user", prompt)
            self.add_to_history("assistant", content)

            return AIResponse(
                content=content,
                provider="gemini",
                model=self.config.model,
                tokens_used=data.get("usageMetadata", {}).get("totalTokenCount", 0),
                latency_ms=latency,
                raw_response=data
            )

        raise Exception(f"Gemini API error: {data}")

    async def generate_code_fix(self, code: str, error: str,
                                 context: Optional[str] = None) -> AIResponse:
        """Generate a code fix using Gemini."""
        system_prompt = """You are an expert software engineer. Fix the code error.
Return ONLY the fixed code without explanations."""

        prompt = f"""Fix this code:

ERROR: {error}

CODE:
```
{code}
```
"""
        if context:
            prompt += f"\nCONTEXT: {context}"

        return await self.generate(prompt, system_prompt)

    async def analyze_code(self, code: str, analysis_type: str = "review") -> AIResponse:
        """Analyze code using Gemini."""
        prompt = f"""Analyze this code for {analysis_type}:

```
{code}
```

Provide: 1) Issues found 2) Suggestions 3) Improved code if applicable"""

        return await self.generate(prompt)


class PerplexityProvider(BaseAIProvider):
    """Perplexity AI API provider - great for research-backed coding."""

    def __init__(self, provider_config: Optional[AIProviderConfig] = None):
        cfg = provider_config or config.get_provider("perplexity")
        super().__init__(cfg)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       stream: bool = False) -> AIResponse:
        """Generate using Perplexity API (OpenAI-compatible)."""
        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.get_history_as_messages())
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.api_base}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()

        latency = (time.time() - start_time) * 1000

        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"]
            self.add_to_history("user", prompt)
            self.add_to_history("assistant", content)

            return AIResponse(
                content=content,
                provider="perplexity",
                model=self.config.model,
                tokens_used=data.get("usage", {}).get("total_tokens", 0),
                latency_ms=latency,
                raw_response=data
            )

        raise Exception(f"Perplexity API error: {data}")

    async def generate_code_fix(self, code: str, error: str,
                                 context: Optional[str] = None) -> AIResponse:
        """Generate a code fix using Perplexity (with web search for latest solutions)."""
        system_prompt = """You are an expert software engineer with access to the latest documentation and solutions.
Fix the code error using the most current best practices and solutions available."""

        prompt = f"""Search for the latest solutions and fix this code error:

ERROR: {error}

CODE:
```
{code}
```

Provide the fixed code with explanations of what you found."""

        return await self.generate(prompt, system_prompt)

    async def analyze_code(self, code: str, analysis_type: str = "review") -> AIResponse:
        """Analyze code using Perplexity (with web search for best practices)."""
        prompt = f"""Search for current best practices and analyze this code for {analysis_type}:

```
{code}
```

Include references to current documentation and best practices you find."""

        return await self.generate(prompt)


class JunieProvider(BaseAIProvider):
    """JetBrains Junie integration via IDE API."""

    def __init__(self, provider_config: Optional[AIProviderConfig] = None):
        # Junie uses IDE integration, not direct API
        cfg = provider_config or AIProviderConfig(
            name="Junie",
            api_base="http://localhost:63342",  # JetBrains IDE API port
            model="junie",
            enabled=True
        )
        super().__init__(cfg)
        self.ide_api_url = "http://localhost:63342/api"

    async def _check_ide_connection(self) -> bool:
        """Check if JetBrains IDE is running and accessible."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ide_api_url}/about",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except:
            return False

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       stream: bool = False) -> AIResponse:
        """Generate using Junie through IDE integration."""
        start_time = time.time()

        # Check IDE connection
        if not await self._check_ide_connection():
            raise Exception("JetBrains IDE not running or Junie not available")

        # Send request to IDE
        payload = {
            "action": "junie.generate",
            "prompt": prompt,
            "context": system_prompt or ""
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ide_api_url}/junie/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.agent.ai_timeout)
                ) as response:
                    data = await response.json()

            latency = (time.time() - start_time) * 1000

            if "result" in data:
                content = data["result"]
                self.add_to_history("user", prompt)
                self.add_to_history("assistant", content)

                return AIResponse(
                    content=content,
                    provider="junie",
                    model="junie",
                    tokens_used=0,
                    latency_ms=latency,
                    raw_response=data
                )

        except Exception as e:
            raise Exception(f"Junie integration error: {e}")

        raise Exception("Junie did not return a valid response")

    async def generate_code_fix(self, code: str, error: str,
                                 context: Optional[str] = None) -> AIResponse:
        """Generate a code fix using Junie."""
        prompt = f"""Fix this code error:
ERROR: {error}
CODE:
```
{code}
```"""
        return await self.generate(prompt, context)

    async def analyze_code(self, code: str, analysis_type: str = "review") -> AIResponse:
        """Analyze code using Junie."""
        prompt = f"""Analyze this code ({analysis_type}):
```
{code}
```"""
        return await self.generate(prompt)


class AIProviderManager:
    """Manages multiple AI providers and coordinates requests."""

    def __init__(self):
        self.providers: Dict[str, BaseAIProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all available providers."""
        enabled = config.get_enabled_providers()

        if "claude" in enabled:
            self.providers["claude"] = ClaudeProvider()

        if "chatgpt" in enabled:
            self.providers["chatgpt"] = ChatGPTProvider()

        if "gemini" in enabled:
            self.providers["gemini"] = GeminiProvider()

        if "perplexity" in enabled:
            self.providers["perplexity"] = PerplexityProvider()

        # Always try to initialize Junie (IDE integration)
        self.providers["junie"] = JunieProvider()

    def get_provider(self, name: str) -> Optional[BaseAIProvider]:
        """Get a provider by name."""
        return self.providers.get(name.lower())

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())

    async def generate_with_fallback(self, prompt: str,
                                      preferred_providers: Optional[List[str]] = None,
                                      system_prompt: Optional[str] = None) -> AIResponse:
        """
        Generate response with automatic fallback to other providers.

        Args:
            prompt: The prompt to send
            preferred_providers: Ordered list of preferred providers
            system_prompt: Optional system prompt

        Returns:
            AIResponse from the first successful provider
        """
        providers_to_try = preferred_providers or list(self.providers.keys())

        errors = []
        for provider_name in providers_to_try:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            try:
                return await provider.generate(prompt, system_prompt)
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)}")
                continue

        raise Exception(f"All providers failed: {'; '.join(errors)}")

    async def generate_parallel(self, prompt: str,
                                 providers: Optional[List[str]] = None,
                                 system_prompt: Optional[str] = None) -> List[AIResponse]:
        """
        Generate responses from multiple providers in parallel.

        Args:
            prompt: The prompt to send
            providers: List of providers to use (default: all)
            system_prompt: Optional system prompt

        Returns:
            List of AIResponses from all successful providers
        """
        providers_to_use = providers or list(self.providers.keys())

        async def get_response(name: str) -> Optional[AIResponse]:
            provider = self.providers.get(name)
            if not provider:
                return None
            try:
                return await provider.generate(prompt, system_prompt)
            except:
                return None

        tasks = [get_response(name) for name in providers_to_use]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

    async def consensus_generate(self, prompt: str,
                                  min_agreement: int = 2,
                                  system_prompt: Optional[str] = None) -> AIResponse:
        """
        Generate responses from multiple providers and return consensus.

        Args:
            prompt: The prompt to send
            min_agreement: Minimum number of similar responses required
            system_prompt: Optional system prompt

        Returns:
            The most common response or first response if no consensus
        """
        responses = await self.generate_parallel(prompt, system_prompt=system_prompt)

        if not responses:
            raise Exception("No providers returned a response")

        if len(responses) == 1:
            return responses[0]

        # Simple consensus: return the longest response (usually most complete)
        # In production, you'd want semantic similarity comparison
        return max(responses, key=lambda r: len(r.content))


# Global provider manager instance
provider_manager = AIProviderManager()

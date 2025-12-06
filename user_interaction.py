"""
User Interaction Module for Coding Agent
Handles prompts for input, resources, and user decisions.
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table


class InteractionType(Enum):
    """Types of user interaction."""
    CONFIRMATION = "confirmation"
    TEXT_INPUT = "text_input"
    CHOICE = "choice"
    FILE_PATH = "file_path"
    RESOURCE_REQUEST = "resource_request"
    CODE_REVIEW = "code_review"
    API_KEY = "api_key"
    CLARIFICATION = "clarification"


@dataclass
class UserResponse:
    """Response from user interaction."""
    interaction_type: InteractionType
    value: Any
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UserInteractionManager:
    """
    Manages all user interactions during agent execution.
    Asks for input, resources, confirmations, and clarifications.
    """

    def __init__(self, auto_mode: bool = False, console: Optional[Console] = None):
        self.console = console or Console()
        self.auto_mode = auto_mode
        self.interaction_history: List[Dict[str, Any]] = []
        self.cached_responses: Dict[str, Any] = {}

    def _record_interaction(self, interaction_type: InteractionType,
                             prompt: str, response: Any):
        """Record an interaction for the journal."""
        self.interaction_history.append({
            "type": interaction_type.value,
            "prompt": prompt,
            "response": str(response)[:200]
        })

    def ask_confirmation(self, message: str, default: bool = True,
                         context: Optional[str] = None) -> bool:
        """
        Ask user for confirmation.

        Args:
            message: Question to ask
            default: Default answer if auto_mode
            context: Additional context to show

        Returns:
            True if confirmed, False otherwise
        """
        if self.auto_mode:
            return default

        if context:
            self.console.print(Panel(context, title="Context", border_style="dim"))

        response = Confirm.ask(f"[bold cyan]?[/bold cyan] {message}", default=default)
        self._record_interaction(InteractionType.CONFIRMATION, message, response)
        return response

    def ask_input(self, prompt: str, default: Optional[str] = None,
                   password: bool = False, required: bool = True) -> Optional[str]:
        """
        Ask user for text input.

        Args:
            prompt: Prompt to display
            default: Default value
            password: Whether to hide input (for API keys)
            required: Whether input is required

        Returns:
            User input or default
        """
        if self.auto_mode and default is not None:
            return default

        if password:
            response = Prompt.ask(f"[bold cyan]?[/bold cyan] {prompt}", password=True)
        else:
            response = Prompt.ask(
                f"[bold cyan]?[/bold cyan] {prompt}",
                default=default if default else ""
            )

        if required and not response:
            self.console.print("[red]This field is required[/red]")
            return self.ask_input(prompt, default, password, required)

        self._record_interaction(InteractionType.TEXT_INPUT, prompt, "[HIDDEN]" if password else response)
        return response

    def ask_choice(self, message: str, choices: List[str],
                    default: int = 0) -> str:
        """
        Ask user to choose from options.

        Args:
            message: Question to ask
            choices: List of choices
            default: Default choice index

        Returns:
            Selected choice
        """
        if self.auto_mode:
            return choices[default]

        self.console.print(f"\n[bold cyan]?[/bold cyan] {message}")

        for i, choice in enumerate(choices):
            marker = "[green]>[/green]" if i == default else " "
            self.console.print(f"  {marker} [{i + 1}] {choice}")

        while True:
            response = Prompt.ask("Enter number", default=str(default + 1))
            try:
                idx = int(response) - 1
                if 0 <= idx < len(choices):
                    self._record_interaction(InteractionType.CHOICE, message, choices[idx])
                    return choices[idx]
            except ValueError:
                pass
            self.console.print(f"[red]Please enter a number between 1 and {len(choices)}[/red]")

    def ask_file_path(self, message: str, must_exist: bool = True,
                       file_type: Optional[str] = None) -> Optional[Path]:
        """
        Ask user for a file path.

        Args:
            message: Prompt message
            must_exist: Whether file must exist
            file_type: Expected file extension

        Returns:
            Path object or None
        """
        while True:
            response = self.ask_input(message, required=False)

            if not response:
                return None

            path = Path(response).expanduser()

            if must_exist and not path.exists():
                self.console.print(f"[red]File not found: {path}[/red]")
                continue

            if file_type and not path.suffix.lower() == f".{file_type.lower()}":
                self.console.print(f"[yellow]Expected .{file_type} file[/yellow]")
                if not self.ask_confirmation("Continue anyway?"):
                    continue

            self._record_interaction(InteractionType.FILE_PATH, message, str(path))
            return path

    def request_resource(self, resource_type: str, description: str,
                          options: Optional[List[str]] = None) -> UserResponse:
        """
        Request a resource from the user.

        Args:
            resource_type: Type of resource needed (api_key, file, config, etc.)
            description: Description of why it's needed
            options: Optional list of ways to provide the resource

        Returns:
            UserResponse with the resource
        """
        self.console.print(Panel(
            f"[bold]Resource Needed:[/bold] {resource_type}\n\n{description}",
            title="Resource Request",
            border_style="yellow"
        ))

        if resource_type == "api_key":
            # Check environment first
            env_var = self.ask_input(
                f"Environment variable name for {resource_type} (or press Enter to input directly)"
            )
            if env_var and os.environ.get(env_var):
                return UserResponse(
                    interaction_type=InteractionType.RESOURCE_REQUEST,
                    value=os.environ[env_var],
                    metadata={"source": "environment", "var_name": env_var}
                )

            # Ask for direct input
            key = self.ask_input(f"Enter {resource_type}", password=True)
            return UserResponse(
                interaction_type=InteractionType.RESOURCE_REQUEST,
                value=key,
                metadata={"source": "direct_input"}
            )

        elif resource_type == "file":
            path = self.ask_file_path(f"Path to {description}")
            if path and path.exists():
                with open(path, 'r') as f:
                    content = f.read()
                return UserResponse(
                    interaction_type=InteractionType.RESOURCE_REQUEST,
                    value=content,
                    metadata={"source": "file", "path": str(path)}
                )

        elif resource_type == "config":
            if options:
                choice = self.ask_choice("Select configuration option:", options)
                return UserResponse(
                    interaction_type=InteractionType.RESOURCE_REQUEST,
                    value=choice,
                    metadata={"source": "choice"}
                )
            else:
                value = self.ask_input(description)
                return UserResponse(
                    interaction_type=InteractionType.RESOURCE_REQUEST,
                    value=value,
                    metadata={"source": "input"}
                )

        # Default: ask for text input
        value = self.ask_input(description)
        return UserResponse(
            interaction_type=InteractionType.RESOURCE_REQUEST,
            value=value,
            metadata={"type": resource_type}
        )

    def request_clarification(self, context: str, question: str,
                               suggestions: Optional[List[str]] = None) -> str:
        """
        Request clarification from user about ambiguous situation.

        Args:
            context: Context of the ambiguity
            question: Question to ask
            suggestions: Optional suggestions to choose from

        Returns:
            User's clarification
        """
        self.console.print(Panel(
            f"[dim]{context}[/dim]\n\n[bold]{question}[/bold]",
            title="Clarification Needed",
            border_style="cyan"
        ))

        if suggestions:
            suggestions.append("Other (specify)")
            choice = self.ask_choice("Select an option:", suggestions)
            if choice == "Other (specify)":
                return self.ask_input("Please specify")
            return choice
        else:
            return self.ask_input("Your answer")

    def review_code_change(self, file_path: str, original: str,
                            modified: str, description: str) -> bool:
        """
        Ask user to review a code change before applying.

        Args:
            file_path: Path to the file being changed
            original: Original code
            modified: Modified code
            description: Description of the change

        Returns:
            True if approved, False otherwise
        """
        from rich.syntax import Syntax
        import difflib

        self.console.print(Panel(
            f"[bold]File:[/bold] {file_path}\n[bold]Change:[/bold] {description}",
            title="Code Change Review",
            border_style="yellow"
        ))

        # Generate diff
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="original",
            tofile="modified"
        )
        diff_text = "".join(diff)

        if diff_text:
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            self.console.print("[dim]No visible changes[/dim]")

        approved = self.ask_confirmation("Apply this change?")
        self._record_interaction(InteractionType.CODE_REVIEW, f"Review: {file_path}", approved)
        return approved

    def request_missing_api_key(self, provider: str) -> Optional[str]:
        """
        Request a missing API key from the user.

        Args:
            provider: Name of the AI provider

        Returns:
            API key or None if skipped
        """
        env_vars = {
            "claude": "ANTHROPIC_API_KEY",
            "chatgpt": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY"
        }

        env_var = env_vars.get(provider.lower(), f"{provider.upper()}_API_KEY")

        self.console.print(Panel(
            f"[bold]Provider:[/bold] {provider}\n"
            f"[bold]Environment Variable:[/bold] {env_var}\n\n"
            f"You can also set this in your shell: export {env_var}=your_key",
            title="API Key Required",
            border_style="yellow"
        ))

        choice = self.ask_choice(
            "How would you like to provide the API key?",
            [
                "Enter it now (will be used for this session only)",
                "Skip this provider",
                "Exit and set environment variable"
            ]
        )

        if choice.startswith("Enter"):
            return self.ask_input(f"Enter {provider} API key", password=True)
        elif choice.startswith("Skip"):
            return None
        else:
            self.console.print(f"\nRun: [bold]export {env_var}=your_key[/bold]")
            raise SystemExit(0)

    def select_providers(self, available: List[str], recommended: List[str] = None) -> List[str]:
        """
        Let user select which AI providers to use.

        Args:
            available: List of available providers
            recommended: List of recommended providers

        Returns:
            List of selected providers
        """
        self.console.print("\n[bold]Available AI Providers:[/bold]\n")

        table = Table(show_header=True)
        table.add_column("#", style="dim")
        table.add_column("Provider")
        table.add_column("Status")
        table.add_column("Recommended")

        for i, provider in enumerate(available, 1):
            status = "[green]Ready[/green]"
            rec = "[cyan]*[/cyan]" if recommended and provider in recommended else ""
            table.add_row(str(i), provider, status, rec)

        self.console.print(table)

        response = self.ask_input(
            "Enter provider numbers to use (comma-separated, or 'all')",
            default="all"
        )

        if response.lower() == "all":
            return available

        try:
            indices = [int(x.strip()) - 1 for x in response.split(",")]
            return [available[i] for i in indices if 0 <= i < len(available)]
        except (ValueError, IndexError):
            self.console.print("[yellow]Invalid selection, using all providers[/yellow]")
            return available

    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """Get the history of all interactions."""
        return self.interaction_history.copy()


class AgentPrompts:
    """
    Pre-defined prompts for common agent interactions.
    Standardizes how the agent asks for things.
    """

    @staticmethod
    def need_test_command(current: str) -> Dict[str, Any]:
        """Prompt for test command configuration."""
        return {
            "type": "clarification",
            "context": f"Current test command: {current}",
            "question": "What command should be used to run tests?",
            "suggestions": [
                "pytest",
                "python -m pytest",
                "npm test",
                "go test ./...",
                current
            ]
        }

    @staticmethod
    def need_build_command(current: str) -> Dict[str, Any]:
        """Prompt for build command configuration."""
        return {
            "type": "clarification",
            "context": f"Current build command: {current}",
            "question": "What command should be used to build/compile?",
            "suggestions": [
                "python -m py_compile",
                "npm run build",
                "go build",
                "make",
                current
            ]
        }

    @staticmethod
    def unclear_error(error: str, file: str) -> Dict[str, Any]:
        """Prompt for clarification on unclear error."""
        return {
            "type": "clarification",
            "context": f"Error in {file}:\n{error[:500]}",
            "question": "Can you provide more context about this error?",
            "suggestions": [
                "This is a known issue, proceed with standard fix",
                "Skip this error for now",
                "Let me explain the expected behavior"
            ]
        }

    @staticmethod
    def multiple_fix_options(file: str, options: List[str]) -> Dict[str, Any]:
        """Prompt when multiple fix options are available."""
        return {
            "type": "choice",
            "context": f"Multiple fix options available for {file}",
            "question": "Which fix approach should be used?",
            "options": options
        }

    @staticmethod
    def confirm_destructive_action(action: str, details: str) -> Dict[str, Any]:
        """Prompt for confirmation of destructive action."""
        return {
            "type": "confirmation",
            "context": details,
            "message": f"This will {action}. Continue?",
            "default": False
        }


# Global interaction manager (can be overridden)
_interaction_manager: Optional[UserInteractionManager] = None


def get_interaction_manager() -> UserInteractionManager:
    """Get or create the global interaction manager."""
    global _interaction_manager
    if _interaction_manager is None:
        _interaction_manager = UserInteractionManager()
    return _interaction_manager


def set_auto_mode(enabled: bool):
    """Enable or disable auto mode (no user prompts)."""
    get_interaction_manager().auto_mode = enabled

"""
Terminal UI for Coding Agent
Rich terminal interface with progress tracking and status display.
"""
import sys
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree
from rich.prompt import Prompt, Confirm

from config import config


class TerminalUI:
    """Rich terminal interface for the coding agent."""

    def __init__(self):
        self.console = Console()
        self.start_time = time.time()

    def clear(self):
        """Clear the terminal."""
        self.console.clear()

    def print_banner(self, version: str = "1.0.0"):
        """Print the agent banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║      ██████╗ ██████╗ ██████╗ ██╗███╗   ██╗ ██████╗                       ║
║     ██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║██╔════╝                       ║
║     ██║     ██║   ██║██║  ██║██║██╔██╗ ██║██║  ███╗                      ║
║     ██║     ██║   ██║██║  ██║██║██║╚██╗██║██║   ██║                      ║
║     ╚██████╗╚██████╔╝██████╔╝██║██║ ╚████║╚██████╔╝                      ║
║      ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝                       ║
║                                                                          ║
║      █████╗  ██████╗ ███████╗███╗   ██╗████████╗                         ║
║     ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝                         ║
║     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║                            ║
║     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║                            ║
║     ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║                            ║
║     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                            ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Fix → Test → Optimize → Journal → Push → Repeat                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
        self.console.print(banner, style="bold cyan")
        self.console.print(f"  Version {version} | Mode: {config.agent.mode}", style="dim")
        self.console.print()

    def print_status_bar(self, providers: List[str], local_ai: bool):
        """Print status bar showing available providers."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="dim")
        table.add_column("Value")

        # AI Providers
        if providers:
            provider_str = " ".join([f"[green]●[/green] {p}" for p in providers])
        else:
            provider_str = "[yellow]● No cloud providers[/yellow]"
        table.add_row("Cloud AI:", provider_str)

        # Local AI
        local_status = "[green]● Available[/green]" if local_ai else "[dim]● Not configured[/dim]"
        table.add_row("Local AI:", local_status)

        # Mode
        mode_styles = {
            "standalone": "[yellow]Standalone[/yellow]",
            "single_ai": "[cyan]Single AI[/cyan]",
            "multi_agent": "[magenta]Multi-Agent[/magenta]"
        }
        table.add_row("Mode:", mode_styles.get(config.agent.mode, config.agent.mode))

        self.console.print(Panel(table, title="Status", border_style="blue"))

    def print_help(self):
        """Print help information."""
        help_text = """
## Commands

| Command | Description |
|---------|-------------|
| `run <target>` | Run agent on file or directory |
| `fix <target>` | Fix issues only |
| `optimize <target>` | Optimize code |
| `test` | Run tests |
| `status` | Show current status |
| `journal` | Show session journal |
| `config` | Show/edit configuration |
| `providers` | List available AI providers |
| `local` | Manage local AI models |
| `git` | Git operations |
| `push` | Commit and push changes |
| `help` | Show this help |
| `quit` | Exit |

## Modes

- **standalone**: No AI, rule-based fixes only
- **single_ai**: Use one AI provider
- **multi_agent**: Use multiple AI agents in parallel

## Examples

```bash
run src/              # Process entire src directory
fix main.py           # Fix issues in main.py
optimize utils.py     # Optimize utils.py
push                  # Commit and push all changes
```
"""
        self.console.print(Markdown(help_text))

    def print_info(self, message: str):
        """Print info message."""
        self.console.print(f"[cyan]ℹ[/cyan] {message}")

    def print_success(self, message: str):
        """Print success message."""
        self.console.print(f"[green]✓[/green] {message}")

    def print_error(self, title: str, message: str):
        """Print error message."""
        self.console.print(f"[red]✗ {title}:[/red] {message}")

    def print_warning(self, message: str):
        """Print warning message."""
        self.console.print(f"[yellow]⚠[/yellow] {message}")

    def get_input(self, prompt: str = "> ") -> str:
        """Get user input."""
        return Prompt.ask(prompt)

    def confirm(self, message: str) -> bool:
        """Get user confirmation."""
        return Confirm.ask(message)


class ProgressTracker:
    """Tracks and displays progress of agent operations."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.start_time = time.time()
        self.tasks: Dict[str, Any] = {}

    def create_progress(self) -> Progress:
        """Create a progress display."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

    def format_elapsed(self) -> str:
        """Format elapsed time."""
        elapsed = time.time() - self.start_time
        return f"{elapsed:.1f}s"


class AgentProgressDisplay:
    """Live display for agent progress."""

    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.start_time = time.time()
        self.current_state = "idle"
        self.iterations = []
        self.current_iteration = 0
        self.issues_found = 0
        self.fixes_applied = 0
        self.tests_passed = 0
        self.tests_failed = 0

    def _generate_layout(self) -> Layout:
        """Generate the display layout."""
        layout = Layout()

        # Header
        header = Panel(
            Text("Coding Agent", style="bold cyan", justify="center"),
            box=None
        )

        # Status section
        status_table = Table(show_header=False, box=None, expand=True)
        status_table.add_column("Label", style="dim", width=15)
        status_table.add_column("Value")

        elapsed = time.time() - self.start_time
        status_table.add_row("Elapsed", f"{elapsed:.1f}s")
        status_table.add_row("Iteration", str(self.current_iteration))
        status_table.add_row("State", self._format_state())
        status_table.add_row("Issues", str(self.issues_found))
        status_table.add_row("Fixes", str(self.fixes_applied))
        status_table.add_row("Tests", f"{self.tests_passed}✓ {self.tests_failed}✗")

        status_panel = Panel(status_table, title="Status", border_style="cyan")

        # Progress section
        progress_content = self._generate_progress_content()
        progress_panel = Panel(progress_content, title="Progress", border_style="green")

        layout.split_column(
            Layout(header, size=3),
            Layout(name="main")
        )
        layout["main"].split_row(
            Layout(status_panel, name="status", ratio=1),
            Layout(progress_panel, name="progress", ratio=2)
        )

        return layout

    def _format_state(self) -> str:
        """Format current state with color."""
        state_styles = {
            "idle": "[dim]Idle[/dim]",
            "analyzing": "[cyan]Analyzing...[/cyan]",
            "fixing": "[yellow]Fixing...[/yellow]",
            "testing": "[blue]Testing...[/blue]",
            "optimizing": "[magenta]Optimizing...[/magenta]",
            "journaling": "[dim]Journaling...[/dim]",
            "pushing": "[cyan]Pushing...[/cyan]",
            "completed": "[green]Completed[/green]",
            "failed": "[red]Failed[/red]"
        }
        return state_styles.get(self.current_state, self.current_state)

    def _generate_progress_content(self) -> str:
        """Generate progress content."""
        lines = []

        # Show pipeline stages
        stages = [
            ("analyze", "Analyze Code"),
            ("fix", "Fix Issues"),
            ("test", "Run Tests"),
            ("optimize", "Optimize"),
            ("journal", "Journal"),
            ("push", "Push")
        ]

        for stage_id, stage_name in stages:
            if stage_id == self.current_state:
                lines.append(f"[cyan]→[/cyan] [bold]{stage_name}[/bold]")
            elif self._stage_completed(stage_id):
                lines.append(f"[green]✓[/green] {stage_name}")
            else:
                lines.append(f"[dim]○ {stage_name}[/dim]")

        return "\n".join(lines)

    def _stage_completed(self, stage: str) -> bool:
        """Check if a stage is completed."""
        stage_order = ["analyze", "fix", "test", "optimize", "journal", "push"]
        current_idx = stage_order.index(self.current_state) if self.current_state in stage_order else -1
        stage_idx = stage_order.index(stage) if stage in stage_order else -1
        return stage_idx < current_idx

    def update(self, state: str, **kwargs):
        """Update display state."""
        self.current_state = state

        if "iteration" in kwargs:
            self.current_iteration = kwargs["iteration"]
        if "issues" in kwargs:
            self.issues_found = kwargs["issues"]
        if "fixes" in kwargs:
            self.fixes_applied = kwargs["fixes"]
        if "tests_passed" in kwargs:
            self.tests_passed = kwargs["tests_passed"]
        if "tests_failed" in kwargs:
            self.tests_failed = kwargs["tests_failed"]

    def run(self, update_callback: Callable):
        """Run the live display."""
        with Live(self._generate_layout(), console=self.console, refresh_per_second=4) as live:
            while self.current_state not in ["completed", "failed"]:
                update_callback(self)
                live.update(self._generate_layout())
                time.sleep(0.25)


class ResultsDisplay:
    """Display for agent results."""

    def __init__(self):
        self.console = Console()

    def show_iteration_result(self, iteration: Dict[str, Any]):
        """Show results of a single iteration."""
        table = Table(title=f"Iteration {iteration.get('iteration', '?')}")

        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        table.add_row("State", iteration.get("state", "unknown"))
        table.add_row("Issues Found", str(len(iteration.get("issues_found", []))))
        table.add_row("Fixes Applied", str(len(iteration.get("fixes_applied", []))))
        table.add_row("Duration", f"{iteration.get('duration_seconds', 0):.2f}s")

        test_result = iteration.get("test_result", {})
        if test_result:
            test_str = f"{test_result.get('passed', 0)} passed, {test_result.get('failed', 0)} failed"
            table.add_row("Tests", test_str)

        self.console.print(table)

    def show_summary(self, summary: Dict[str, Any]):
        """Show summary of all iterations."""
        panel_content = []

        panel_content.append(f"[bold]Iterations:[/bold] {summary.get('iterations', 0)}")
        panel_content.append(f"[bold]Issues Found:[/bold] {summary.get('total_issues_found', 0)}")
        panel_content.append(f"[bold]Fixes Applied:[/bold] {summary.get('total_fixes_applied', 0)}")
        panel_content.append(f"[bold]Optimizations:[/bold] {summary.get('total_optimizations', 0)}")
        panel_content.append(f"[bold]Duration:[/bold] {summary.get('total_duration_seconds', 0):.2f}s")

        success = summary.get('final_success', False)
        status = "[green]SUCCESS[/green]" if success else "[red]FAILED[/red]"
        panel_content.append(f"[bold]Status:[/bold] {status}")

        self.console.print(Panel(
            "\n".join(panel_content),
            title="Agent Summary",
            border_style="green" if success else "red"
        ))

    def show_issues(self, issues: List[Dict[str, Any]]):
        """Display a list of issues."""
        if not issues:
            self.console.print("[dim]No issues found[/dim]")
            return

        tree = Tree("[bold]Issues Found[/bold]")

        for issue in issues:
            file_branch = tree.add(f"[cyan]{issue.get('file_path', 'unknown')}[/cyan]")
            file_branch.add(f"Line {issue.get('line_number', '?')}: {issue.get('message', '')}")

        self.console.print(tree)

    def show_fixes(self, fixes: List[Dict[str, Any]]):
        """Display applied fixes."""
        if not fixes:
            self.console.print("[dim]No fixes applied[/dim]")
            return

        table = Table(title="Fixes Applied")
        table.add_column("File", style="cyan")
        table.add_column("Issue")
        table.add_column("Status")

        for fix in fixes:
            status = "[green]✓[/green]" if fix.get("success") else "[red]✗[/red]"
            table.add_row(
                fix.get("file", "unknown"),
                fix.get("issue", "")[:50] + "...",
                status
            )

        self.console.print(table)

    def show_code_diff(self, original: str, fixed: str, file_path: str = ""):
        """Show code diff."""
        self.console.print(f"\n[bold]Changes to {file_path}:[/bold]")

        import difflib
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile="original",
            tofile="fixed"
        )

        diff_text = "".join(diff)
        if diff_text:
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            self.console.print("[dim]No changes[/dim]")


class InteractiveMode:
    """Interactive command mode for the agent."""

    def __init__(self):
        self.ui = TerminalUI()
        self.results = ResultsDisplay()
        self.running = True

    def run(self, run_command: Callable):
        """Run interactive mode."""
        self.ui.clear()
        self.ui.print_banner()

        from ai_providers import provider_manager
        from local_ai import local_ai_manager

        providers = provider_manager.get_available_providers()
        local_available = local_ai_manager.backend is not None

        self.ui.print_status_bar(providers, local_available)
        self.ui.console.print()

        while self.running:
            try:
                command = self.ui.get_input("[bold cyan]agent>[/bold cyan] ")
                self._handle_command(command, run_command)
            except KeyboardInterrupt:
                self.ui.console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            except Exception as e:
                self.ui.print_error("Error", str(e))

    def _handle_command(self, command: str, run_command: Callable):
        """Handle a command."""
        parts = command.strip().split()
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ["quit", "exit", "q"]:
            self.running = False
            self.ui.print_info("Goodbye!")

        elif cmd == "help":
            self.ui.print_help()

        elif cmd == "run":
            if args:
                run_command("run", args[0])
            else:
                self.ui.print_error("Usage", "run <target>")

        elif cmd == "fix":
            if args:
                run_command("fix", args[0])
            else:
                self.ui.print_error("Usage", "fix <target>")

        elif cmd == "optimize":
            if args:
                run_command("optimize", args[0])
            else:
                self.ui.print_error("Usage", "optimize <target>")

        elif cmd == "test":
            run_command("test", None)

        elif cmd == "status":
            run_command("status", None)

        elif cmd == "journal":
            run_command("journal", None)

        elif cmd == "providers":
            run_command("providers", None)

        elif cmd == "push":
            run_command("push", None)

        elif cmd == "config":
            run_command("config", args[0] if args else None)

        else:
            self.ui.print_warning(f"Unknown command: {cmd}. Type 'help' for available commands.")

#!/usr/bin/env python3
"""
Coding Agent - Autonomous Code Optimization System
Main entry point and CLI interface.

Usage:
    python main.py                      # Interactive mode
    python main.py run <target>         # Run agent on target
    python main.py fix <file>           # Fix issues in file
    python main.py optimize <file>      # Optimize file
    python main.py test                 # Run tests
    python main.py --help               # Show help
"""
import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

from config import config
from ai_providers import provider_manager
from local_ai import local_ai_manager
from agent_engine import AgentEngine, run_agent
from orchestrator import AgentOrchestrator, run_multi_agent
from journal import get_journal, new_session
from git_integration import GitManager, GitHubManager, GitHubAccountManager, create_and_push_repos
from terminal_ui import TerminalUI, ResultsDisplay, InteractiveMode, AgentProgressDisplay
from code_search import CodeSearchManager, search_similar_code


VERSION = "1.0.0"


class CodingAgentCLI:
    """Main CLI application for the Coding Agent."""

    def __init__(self):
        self.ui = TerminalUI()
        self.results = ResultsDisplay()
        self.journal = None
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.ui.console.print("\n[yellow]Interrupted. Cleaning up...[/yellow]")
            if self.journal:
                self.journal.close_session()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def run_interactive(self):
        """Run in interactive mode."""
        self.journal = new_session()

        def run_command(cmd: str, target: Optional[str]):
            asyncio.run(self._execute_command(cmd, target))

        interactive = InteractiveMode()
        interactive.run(run_command)

        if self.journal:
            summary = self.journal.close_session()
            self.ui.print_info(f"Session saved to: {summary['journal_file']}")

    async def _execute_command(self, cmd: str, target: Optional[str]):
        """Execute a command."""
        if cmd == "run":
            await self._run_agent(target, mode="full")

        elif cmd == "fix":
            await self._run_agent(target, mode="fix")

        elif cmd == "optimize":
            await self._run_agent(target, mode="optimize")

        elif cmd == "test":
            await self._run_tests()

        elif cmd == "status":
            self._show_status()

        elif cmd == "journal":
            self._show_journal()

        elif cmd == "providers":
            self._show_providers()

        elif cmd == "push":
            await self._push_changes()

        elif cmd == "config":
            self._show_config(target)

        elif cmd == "search":
            await self._search_code(target)

    async def _run_agent(self, target: str, mode: str = "fix"):
        """Run the agent on a target."""
        if not target:
            self.ui.print_error("Error", "No target specified")
            return

        path = Path(target)
        if not path.exists():
            self.ui.print_error("Error", f"Target not found: {target}")
            return

        self.ui.console.print(f"\n[bold]Running agent on:[/bold] {target}")
        self.ui.console.print(f"[dim]Mode: {mode}[/dim]\n")

        # Record in journal
        if self.journal:
            self.journal.record_iteration_start(1, target, mode)

        def progress_callback(state: str, message: str = ""):
            elapsed = self._format_elapsed()
            self.ui.console.print(f"  [{elapsed}] {state}: {message}")

        # Determine execution mode
        if config.agent.mode == "multi_agent":
            # Multi-agent mode
            self.ui.print_info("Using multi-agent orchestration...")
            results = await run_multi_agent(target, "parallel", progress_callback)
            self.ui.print_success("Multi-agent run complete")
            self.results.show_summary(results.get("status", {}))

        else:
            # Single agent or standalone mode
            engine = AgentEngine(progress_callback=progress_callback)

            if config.agent.mode == "standalone":
                iterations = await engine.run_standalone(target)
            else:
                iterations = await engine.run(target, mode)

            summary = engine.get_summary()
            self.results.show_summary(summary)

            # Record in journal
            if self.journal:
                self.journal.record_iteration_end(
                    1, summary["final_success"],
                    summary["total_duration_seconds"], summary
                )

    async def _run_tests(self):
        """Run tests."""
        from agent_engine import TestRunner

        self.ui.print_info("Running tests...")

        runner = TestRunner()
        result = await runner.run_tests()

        if result.success:
            self.ui.print_success(
                f"Tests passed: {result.passed}/{result.total_tests} in {result.duration_seconds:.2f}s"
            )
        else:
            self.ui.print_error(
                "Tests failed",
                f"{result.failed} failed, {result.errors} errors"
            )

        if result.output and config.agent.verbose:
            self.ui.console.print("\n[dim]Test output:[/dim]")
            self.ui.console.print(result.output[:2000])

        # Record in journal
        if self.journal:
            self.journal.record_test_run(
                config.agent.test_command,
                {
                    "success": result.success,
                    "passed": result.passed,
                    "failed": result.failed,
                    "duration_seconds": result.duration_seconds
                }
            )

    def _show_status(self):
        """Show current status."""
        self.ui.console.print("\n[bold]Current Status[/bold]\n")

        # Show provider status
        providers = provider_manager.get_available_providers()
        self.ui.console.print(f"[cyan]AI Providers:[/cyan] {', '.join(providers) if providers else 'None configured'}")

        # Show local AI status
        local_status = local_ai_manager.get_status()
        self.ui.console.print(f"[cyan]Local AI:[/cyan] {'Enabled' if local_status['enabled'] else 'Disabled'}")
        if local_status.get('model_loaded'):
            self.ui.console.print(f"  Model: {local_status['current_model']}")

        # Show git status
        try:
            git = GitManager()
            status = git.get_status()
            self.ui.console.print(f"[cyan]Git Branch:[/cyan] {status.branch}")
            self.ui.console.print(f"[cyan]Clean:[/cyan] {'Yes' if status.is_clean else 'No'}")
            if not status.is_clean:
                self.ui.console.print(f"  Staged: {len(status.staged_files)}")
                self.ui.console.print(f"  Modified: {len(status.unstaged_files)}")
                self.ui.console.print(f"  Untracked: {len(status.untracked_files)}")
        except Exception as e:
            self.ui.console.print(f"[cyan]Git:[/cyan] Not a git repository")

        # Show journal status
        if self.journal:
            summary = self.journal.get_session_summary()
            self.ui.console.print(f"\n[cyan]Session:[/cyan] {summary['session_id']}")
            self.ui.console.print(f"  Duration: {summary['duration']}")
            self.ui.console.print(f"  AI Calls: {summary['total_ai_calls']}")
            self.ui.console.print(f"  Tokens: {summary['total_tokens_used']}")

    def _show_journal(self):
        """Show journal summary."""
        if not self.journal:
            self.ui.print_warning("No active session")
            return

        summary = self.journal.get_session_summary()

        self.ui.console.print("\n[bold]Session Journal[/bold]\n")
        self.ui.console.print(f"Session ID: {summary['session_id']}")
        self.ui.console.print(f"Duration: {summary['duration']}")
        self.ui.console.print(f"Iterations: {summary['total_iterations']}")
        self.ui.console.print(f"Fixes Attempted: {summary['total_fixes_attempted']}")
        self.ui.console.print(f"Successful Fixes: {summary['successful_fixes']}")
        self.ui.console.print(f"Test Runs: {summary['total_test_runs']}")
        self.ui.console.print(f"AI Calls: {summary['total_ai_calls']}")
        self.ui.console.print(f"Tokens Used: {summary['total_tokens_used']}")
        self.ui.console.print(f"Errors: {summary['errors_count']}")
        self.ui.console.print(f"\nJournal: {summary['journal_file']}")

    def _show_providers(self):
        """Show available AI providers."""
        self.ui.console.print("\n[bold]AI Providers[/bold]\n")

        for name, provider_config in config.providers.items():
            has_key = bool(provider_config.api_key)
            status = "[green]●[/green] Ready" if has_key else "[red]○[/red] No API key"
            self.ui.console.print(f"  {provider_config.name}: {status}")
            if has_key:
                self.ui.console.print(f"    Model: {provider_config.model}")

        # Local AI
        local_status = local_ai_manager.get_status()
        if local_status["enabled"]:
            self.ui.console.print(f"\n  Local AI ({local_status['backend']}):")
            if local_status["model_loaded"]:
                self.ui.console.print(f"    [green]●[/green] {local_status['current_model']}")
            else:
                self.ui.console.print(f"    [yellow]○[/yellow] No model loaded")
        else:
            self.ui.console.print("\n  Local AI: [dim]Disabled[/dim]")

    async def _push_changes(self):
        """Commit and push changes."""
        git = GitManager()
        status = git.get_status()

        if status.is_clean:
            self.ui.print_info("No changes to commit")
            return

        self.ui.console.print("\n[bold]Changes to commit:[/bold]")
        for f in status.staged_files + status.unstaged_files:
            self.ui.console.print(f"  {f}")

        if not self.ui.confirm("Commit and push these changes?"):
            return

        # Get commit message
        message = self.ui.get_input("Commit message: ")
        if not message:
            message = "[Agent] Automated code improvements"

        # Stage all changes
        git.stage_all()

        # Commit
        success, result = git.commit(message)
        if success:
            self.ui.print_success(f"Committed: {result}")

            # Push
            push_success, push_result = git.push()
            if push_success:
                self.ui.print_success("Pushed to remote")
            else:
                self.ui.print_error("Push failed", push_result)

            # Record in journal
            if self.journal:
                self.journal.record_git_action("commit_push", {
                    "commit_hash": result,
                    "message": message,
                    "push_success": push_success
                })
        else:
            self.ui.print_error("Commit failed", result)

    async def _search_code(self, query: str):
        """Search for similar code across web sources."""
        if not query:
            self.ui.print_error("Error", "No search query specified")
            return

        self.ui.console.print(f"\n[bold]Searching for:[/bold] {query}\n")

        from rich.table import Table
        from rich.panel import Panel

        results = await search_similar_code(query)

        if not results:
            self.ui.print_warning("No results found")
            return

        # Group results by source
        by_source = {}
        for r in results:
            if r.source not in by_source:
                by_source[r.source] = []
            by_source[r.source].append(r)

        for source, source_results in by_source.items():
            self.ui.console.print(f"\n[bold cyan]{source.upper()}[/bold cyan]")

            table = Table(show_header=True, header_style="bold")
            table.add_column("#", width=3)
            table.add_column("Title", width=50)
            table.add_column("Language", width=10)
            table.add_column("Relevance", width=10)

            for i, r in enumerate(source_results[:5], 1):
                table.add_row(
                    str(i),
                    r.title[:47] + "..." if len(r.title) > 50 else r.title,
                    r.language or "-",
                    f"{r.relevance:.0%}" if r.relevance else "-"
                )

            self.ui.console.print(table)

            # Show URLs
            for i, r in enumerate(source_results[:5], 1):
                self.ui.console.print(f"  [{i}] [dim]{r.url}[/dim]")

        self.ui.console.print(f"\n[green]Found {len(results)} total results[/green]")

        # Record in journal
        if self.journal:
            self.journal.record_ai_call(
                "code_search",
                {"query": query, "results_count": len(results)},
                {"sources": list(by_source.keys())}
            )

    def _show_config(self, key: Optional[str] = None):
        """Show or modify configuration."""
        self.ui.console.print("\n[bold]Configuration[/bold]\n")

        self.ui.console.print("[cyan]Agent Settings:[/cyan]")
        self.ui.console.print(f"  Mode: {config.agent.mode}")
        self.ui.console.print(f"  Max Iterations: {config.agent.max_iterations}")
        self.ui.console.print(f"  Test Command: {config.agent.test_command}")
        self.ui.console.print(f"  Build Command: {config.agent.build_command}")

        self.ui.console.print("\n[cyan]Local AI:[/cyan]")
        self.ui.console.print(f"  Enabled: {config.local_ai.enabled}")
        self.ui.console.print(f"  Backend: {config.local_ai.backend}")
        self.ui.console.print(f"  Model: {config.local_ai.model_name}")

        self.ui.console.print("\n[cyan]Git:[/cyan]")
        self.ui.console.print(f"  Auto Commit: {config.git.auto_commit}")
        self.ui.console.print(f"  Auto Push: {config.git.auto_push}")
        self.ui.console.print(f"  GitHub Accounts: {', '.join(config.git.github_accounts)}")

    def _format_elapsed(self) -> str:
        """Format elapsed time since start."""
        import time
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        elapsed = time.time() - self._start_time
        return f"{elapsed:6.1f}s"

    async def run_single_command(self, args):
        """Run a single command from CLI args."""
        self.journal = new_session()
        self._start_time = __import__('time').time()

        try:
            if args.command == "run":
                await self._run_agent(args.target, mode="full")

            elif args.command == "fix":
                await self._run_agent(args.target, mode="fix")

            elif args.command == "optimize":
                await self._run_agent(args.target, mode="optimize")

            elif args.command == "test":
                await self._run_tests()

            elif args.command == "status":
                self._show_status()

            elif args.command == "providers":
                self._show_providers()

            elif args.command == "push":
                await self._push_changes()

            elif args.command == "create-repos":
                await self._create_github_repos(args)

            elif args.command == "search":
                await self._search_code(args.query)

        finally:
            if self.journal:
                summary = self.journal.close_session()
                if args.verbose:
                    self.ui.print_info(f"Session: {summary['journal_file']}")

    async def _create_github_repos(self, args):
        """Create GitHub repositories on configured accounts."""
        repo_name = args.name or "coding-agent"
        description = args.description or "Autonomous code optimization agent"

        self.ui.print_info(f"Creating repository '{repo_name}' on configured accounts...")

        results = await create_and_push_repos(
            repo_name=repo_name,
            description=description,
            private=args.private if hasattr(args, 'private') else False
        )

        for account, (success, result) in results.get("repo_creation", {}).items():
            if success:
                self.ui.print_success(f"{account}: {result}")
            else:
                self.ui.print_error(account, result)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Coding Agent - Autonomous Code Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Interactive mode
  python main.py run src/             # Process src directory
  python main.py fix main.py          # Fix issues in main.py
  python main.py optimize utils.py    # Optimize utils.py
  python main.py test                 # Run tests
  python main.py push                 # Commit and push changes
  python main.py create-repos         # Create GitHub repos on nebulai13 and qwitch13
  python main.py search "async await" # Search for similar code patterns

Environment Variables:
  ANTHROPIC_API_KEY    - Claude API key
  OPENAI_API_KEY       - ChatGPT API key
  GOOGLE_API_KEY       - Gemini API key
  PERPLEXITY_API_KEY   - Perplexity API key

Modes:
  standalone   - No AI, rule-based fixes only
  single_ai    - Use one AI provider
  multi_agent  - Use multiple AI agents in parallel
        """
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"Coding Agent v{VERSION}"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["standalone", "single_ai", "multi_agent"],
        help="Operation mode"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run agent on target")
    run_parser.add_argument("target", help="File or directory to process")

    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Fix issues in target")
    fix_parser.add_argument("target", help="File or directory to fix")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize target")
    opt_parser.add_argument("target", help="File or directory to optimize")

    # Test command
    subparsers.add_parser("test", help="Run tests")

    # Status command
    subparsers.add_parser("status", help="Show status")

    # Providers command
    subparsers.add_parser("providers", help="Show AI providers")

    # Push command
    subparsers.add_parser("push", help="Commit and push changes")

    # Create repos command
    repos_parser = subparsers.add_parser("create-repos", help="Create GitHub repositories")
    repos_parser.add_argument("--name", "-n", default="coding-agent", help="Repository name")
    repos_parser.add_argument("--description", "-d", help="Repository description")
    repos_parser.add_argument("--private", "-p", action="store_true", help="Create private repos")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar code on the web")
    search_parser.add_argument("query", help="Search query or code snippet")
    search_parser.add_argument("--language", "-l", help="Filter by programming language")
    search_parser.add_argument("--source", "-s", choices=["github", "stackoverflow", "google", "all"],
                               default="all", help="Search source")

    args = parser.parse_args()

    # Apply mode override
    if args.mode:
        config.agent.mode = args.mode

    if args.verbose:
        config.agent.verbose = True

    # Create CLI instance
    cli = CodingAgentCLI()

    # Run appropriate mode
    if args.command:
        asyncio.run(cli.run_single_command(args))
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()

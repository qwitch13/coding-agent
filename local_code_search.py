"""
Local Code Search Module for Coding Agent
Search local directories for code patterns and extract to current project.
"""
import os
import re
import json
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from fnmatch import fnmatch

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


CONFIG_FILE = ".local_search_config.json"
DEFAULT_EXTENSIONS = {
    "python": [".py", ".pyx", ".pyi"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "kotlin": [".kt", ".kts"],
    "go": [".go"],
    "rust": [".rs"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "shell": [".sh", ".bash", ".zsh"],
    "sql": [".sql"],
    "html": [".html", ".htm"],
    "css": [".css", ".scss", ".sass", ".less"],
    "yaml": [".yaml", ".yml"],
    "json": [".json"],
    "markdown": [".md", ".markdown"],
}

IGNORE_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv", "venv",
    ".env", "env", "dist", "build", ".idea", ".vscode", ".cache",
    "target", "out", ".gradle", ".maven", "vendor", "Pods",
}

IGNORE_FILES = {
    ".DS_Store", "Thumbs.db", ".gitignore", ".dockerignore",
    "package-lock.json", "yarn.lock", "Cargo.lock", "poetry.lock",
}


@dataclass
class LocalSearchResult:
    """Result from local code search."""
    file_path: str
    relative_path: str
    source_dir: str
    matches: List[Dict[str, Any]] = field(default_factory=list)
    language: str = ""
    file_size: int = 0
    modified_time: float = 0.0
    relevance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchDirectory:
    """Configured search directory."""
    path: str
    name: str = ""
    enabled: bool = True
    languages: List[str] = field(default_factory=list)
    added_at: str = ""
    last_searched: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = Path(self.path).name
        if not self.added_at:
            self.added_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalSearchConfig:
    """Manages local search configuration."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.cwd() / CONFIG_FILE
        self.directories: List[SearchDirectory] = []
        self.default_languages: List[str] = []
        self.max_file_size: int = 1024 * 1024  # 1MB
        self.max_results: int = 100
        self.load()

    def load(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                self.directories = [
                    SearchDirectory(**d) for d in data.get("directories", [])
                ]
                self.default_languages = data.get("default_languages", [])
                self.max_file_size = data.get("max_file_size", self.max_file_size)
                self.max_results = data.get("max_results", self.max_results)
            except (json.JSONDecodeError, KeyError) as e:
                Console().print(f"[yellow]Warning: Could not load config: {e}[/yellow]")

    def save(self):
        """Save configuration to file."""
        data = {
            "directories": [d.to_dict() for d in self.directories],
            "default_languages": self.default_languages,
            "max_file_size": self.max_file_size,
            "max_results": self.max_results,
        }
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_directory(self, path: str, name: str = "", languages: List[str] = None) -> bool:
        """Add a search directory."""
        abs_path = str(Path(path).expanduser().resolve())

        if not Path(abs_path).exists():
            return False

        # Check for duplicates
        for d in self.directories:
            if d.path == abs_path:
                return False

        self.directories.append(SearchDirectory(
            path=abs_path,
            name=name or Path(abs_path).name,
            languages=languages or [],
        ))
        self.save()
        return True

    def remove_directory(self, path_or_name: str) -> bool:
        """Remove a search directory."""
        for i, d in enumerate(self.directories):
            if d.path == path_or_name or d.name == path_or_name:
                del self.directories[i]
                self.save()
                return True
        return False

    def get_enabled_directories(self) -> List[SearchDirectory]:
        """Get all enabled search directories."""
        return [d for d in self.directories if d.enabled and Path(d.path).exists()]

    def toggle_directory(self, path_or_name: str) -> bool:
        """Toggle a directory's enabled state."""
        for d in self.directories:
            if d.path == path_or_name or d.name == path_or_name:
                d.enabled = not d.enabled
                self.save()
                return True
        return False


class LocalCodeSearcher:
    """Searches local directories for code patterns."""

    def __init__(self, config: Optional[LocalSearchConfig] = None):
        self.config = config or LocalSearchConfig()
        self.console = Console()
        self._file_cache: Dict[str, str] = {}

    def _get_extensions(self, languages: List[str] = None) -> Set[str]:
        """Get file extensions to search."""
        if not languages:
            languages = self.config.default_languages

        if not languages:
            # Return all known extensions
            extensions = set()
            for exts in DEFAULT_EXTENSIONS.values():
                extensions.update(exts)
            return extensions

        extensions = set()
        for lang in languages:
            lang_lower = lang.lower()
            if lang_lower in DEFAULT_EXTENSIONS:
                extensions.update(DEFAULT_EXTENSIONS[lang_lower])
            elif lang_lower.startswith("."):
                extensions.add(lang_lower)
        return extensions

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        name = path.name

        if path.is_dir():
            return name in IGNORE_DIRS or name.startswith(".")

        if name in IGNORE_FILES:
            return True

        try:
            if path.stat().st_size > self.config.max_file_size:
                return True
        except OSError:
            return True

        return False

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext = file_path.suffix.lower()
        for lang, extensions in DEFAULT_EXTENSIONS.items():
            if ext in extensions:
                return lang
        return ""

    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file contents with caching."""
        path_str = str(file_path)

        if path_str in self._file_cache:
            return self._file_cache[path_str]

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self._file_cache[path_str] = content
            return content
        except (OSError, UnicodeDecodeError):
            return None

    def _calculate_relevance(self, matches: List[Dict], file_path: Path, query: str) -> float:
        """Calculate relevance score for a file."""
        if not matches:
            return 0.0

        score = 0.0

        # Number of matches
        score += min(len(matches) * 0.1, 0.3)

        # Exact match bonus
        for match in matches:
            if query.lower() in match.get("line", "").lower():
                score += 0.2
                break

        # Filename match bonus
        if query.lower() in file_path.name.lower():
            score += 0.3

        # Recent modification bonus
        try:
            mtime = file_path.stat().st_mtime
            days_old = (datetime.now().timestamp() - mtime) / 86400
            if days_old < 7:
                score += 0.2
            elif days_old < 30:
                score += 0.1
        except OSError:
            pass

        return min(score, 1.0)

    def search(self, query: str, languages: List[str] = None,
               directories: List[str] = None, regex: bool = False,
               context_lines: int = 2) -> List[LocalSearchResult]:
        """
        Search for code matching query in configured directories.

        Args:
            query: Search pattern (string or regex)
            languages: Filter by programming languages
            directories: Specific directories to search (uses configured if None)
            regex: Whether query is a regex pattern
            context_lines: Lines of context around matches

        Returns:
            List of LocalSearchResult
        """
        results = []
        extensions = self._get_extensions(languages)

        # Get directories to search
        if directories:
            search_dirs = [SearchDirectory(path=d) for d in directories if Path(d).exists()]
        else:
            search_dirs = self.config.get_enabled_directories()

        if not search_dirs:
            self.console.print("[yellow]No search directories configured. Use 'local-search add <path>'[/yellow]")
            return results

        # Compile pattern
        if regex:
            try:
                pattern = re.compile(query, re.IGNORECASE)
            except re.error as e:
                self.console.print(f"[red]Invalid regex: {e}[/red]")
                return results
        else:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        files_searched = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Searching...", total=None)

            for search_dir in search_dirs:
                dir_path = Path(search_dir.path)
                progress.update(task, description=f"Searching {search_dir.name}...")

                for root, dirs, files in os.walk(dir_path):
                    # Filter directories
                    dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]

                    for filename in files:
                        file_path = Path(root) / filename

                        if self._should_ignore(file_path):
                            continue

                        if extensions and file_path.suffix.lower() not in extensions:
                            continue

                        files_searched += 1
                        content = self._read_file(file_path)

                        if content is None:
                            continue

                        # Search for matches
                        matches = []
                        lines = content.splitlines()

                        for i, line in enumerate(lines):
                            if pattern.search(line):
                                # Get context
                                start = max(0, i - context_lines)
                                end = min(len(lines), i + context_lines + 1)
                                context = lines[start:end]

                                matches.append({
                                    "line_number": i + 1,
                                    "line": line.strip(),
                                    "context": context,
                                    "context_start": start + 1,
                                })

                        if matches:
                            try:
                                stat = file_path.stat()
                                file_size = stat.st_size
                                modified_time = stat.st_mtime
                            except OSError:
                                file_size = 0
                                modified_time = 0

                            result = LocalSearchResult(
                                file_path=str(file_path),
                                relative_path=str(file_path.relative_to(dir_path)),
                                source_dir=search_dir.name,
                                matches=matches,
                                language=self._detect_language(file_path),
                                file_size=file_size,
                                modified_time=modified_time,
                            )
                            result.relevance = self._calculate_relevance(matches, file_path, query)
                            results.append(result)

                            if len(results) >= self.config.max_results:
                                break

                    if len(results) >= self.config.max_results:
                        break

                # Update last searched time
                search_dir.last_searched = datetime.now().isoformat()

            self.config.save()

        # Sort by relevance
        results.sort(key=lambda r: r.relevance, reverse=True)

        self.console.print(f"[dim]Searched {files_searched} files[/dim]")

        return results

    def search_similar(self, code_snippet: str, languages: List[str] = None,
                       min_similarity: float = 0.5) -> List[LocalSearchResult]:
        """
        Search for code similar to the provided snippet.

        Args:
            code_snippet: Code to find similar matches for
            languages: Filter by programming languages
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of LocalSearchResult with similarity scores
        """
        # Extract key tokens from snippet
        tokens = set(re.findall(r'\b\w+\b', code_snippet.lower()))
        tokens -= {'the', 'a', 'an', 'is', 'are', 'be', 'to', 'of', 'and', 'or', 'in', 'for', 'if', 'else'}

        if not tokens:
            return []

        # Search for files containing any of the tokens
        results = []

        for token in list(tokens)[:5]:  # Limit to top 5 tokens
            token_results = self.search(token, languages=languages)
            results.extend(token_results)

        # Deduplicate and calculate similarity
        seen_files = {}
        for result in results:
            if result.file_path not in seen_files:
                # Calculate token overlap similarity
                content = self._read_file(Path(result.file_path))
                if content:
                    file_tokens = set(re.findall(r'\b\w+\b', content.lower()))
                    overlap = len(tokens & file_tokens)
                    similarity = overlap / len(tokens) if tokens else 0

                    if similarity >= min_similarity:
                        result.relevance = similarity
                        seen_files[result.file_path] = result

        final_results = list(seen_files.values())
        final_results.sort(key=lambda r: r.relevance, reverse=True)

        return final_results[:self.config.max_results]

    def extract_to_current(self, result: LocalSearchResult,
                           destination: str = None,
                           preserve_structure: bool = True) -> Tuple[bool, str]:
        """
        Extract/copy a found file to the current project.

        Args:
            result: Search result to extract
            destination: Destination path (defaults to current dir)
            preserve_structure: Whether to preserve directory structure

        Returns:
            (success, message)
        """
        source = Path(result.file_path)

        if not source.exists():
            return False, f"Source file not found: {source}"

        if destination:
            dest_base = Path(destination)
        else:
            dest_base = Path.cwd()

        if preserve_structure:
            # Create subdirectory based on source
            dest = dest_base / "extracted" / result.source_dir / result.relative_path
        else:
            dest = dest_base / "extracted" / source.name

        # Handle name conflicts
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = dest.parent / f"{stem}_{counter}{suffix}"
                counter += 1

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            return True, str(dest)
        except (OSError, shutil.Error) as e:
            return False, str(e)

    def extract_multiple(self, results: List[LocalSearchResult],
                        destination: str = None) -> Dict[str, Any]:
        """
        Extract multiple files to current project.

        Args:
            results: List of search results to extract
            destination: Destination directory

        Returns:
            Summary of extraction results
        """
        summary = {
            "total": len(results),
            "success": 0,
            "failed": 0,
            "extracted_files": [],
            "errors": [],
        }

        for result in results:
            success, message = self.extract_to_current(result, destination)
            if success:
                summary["success"] += 1
                summary["extracted_files"].append(message)
            else:
                summary["failed"] += 1
                summary["errors"].append(f"{result.file_path}: {message}")

        return summary


class LocalSearchManager:
    """High-level manager for local code search operations."""

    def __init__(self):
        self.config = LocalSearchConfig()
        self.searcher = LocalCodeSearcher(self.config)
        self.console = Console()

    def add_directory(self, path: str, name: str = "", languages: List[str] = None) -> bool:
        """Add a directory to search."""
        success = self.config.add_directory(path, name, languages)
        if success:
            self.console.print(f"[green]Added directory: {path}[/green]")
        else:
            expanded = Path(path).expanduser().resolve()
            if not expanded.exists():
                self.console.print(f"[red]Directory not found: {path}[/red]")
            else:
                self.console.print(f"[yellow]Directory already configured: {path}[/yellow]")
        return success

    def remove_directory(self, path_or_name: str) -> bool:
        """Remove a directory from search."""
        success = self.config.remove_directory(path_or_name)
        if success:
            self.console.print(f"[green]Removed: {path_or_name}[/green]")
        else:
            self.console.print(f"[yellow]Not found: {path_or_name}[/yellow]")
        return success

    def list_directories(self):
        """List configured search directories."""
        dirs = self.config.directories

        if not dirs:
            self.console.print("[yellow]No directories configured[/yellow]")
            self.console.print("[dim]Add with: local-search add <path>[/dim]")
            return

        table = Table(title="Search Directories", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Path")
        table.add_column("Status")
        table.add_column("Languages")
        table.add_column("Last Searched")

        for d in dirs:
            exists = Path(d.path).exists()
            status = "[green]enabled[/green]" if d.enabled and exists else "[red]disabled[/red]"
            if not exists:
                status = "[red]not found[/red]"

            languages = ", ".join(d.languages) if d.languages else "[dim]all[/dim]"
            last_searched = d.last_searched[:10] if d.last_searched else "[dim]never[/dim]"

            table.add_row(d.name, d.path, status, languages, last_searched)

        self.console.print(table)

    def search(self, query: str, languages: List[str] = None,
               regex: bool = False) -> List[LocalSearchResult]:
        """Search for code matching query."""
        return self.searcher.search(query, languages=languages, regex=regex)

    def search_and_display(self, query: str, languages: List[str] = None,
                          regex: bool = False, show_context: bool = True) -> List[LocalSearchResult]:
        """Search and display results in a formatted table."""
        results = self.search(query, languages, regex)

        if not results:
            self.console.print("[yellow]No matches found[/yellow]")
            return results

        self.console.print(f"\n[bold green]Found {len(results)} matching files[/bold green]\n")

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", width=3)
        table.add_column("Source", width=15)
        table.add_column("File", width=40)
        table.add_column("Lang", width=8)
        table.add_column("Matches", width=8)
        table.add_column("Score", width=6)

        for i, result in enumerate(results[:20], 1):
            table.add_row(
                str(i),
                result.source_dir,
                result.relative_path[:37] + "..." if len(result.relative_path) > 40 else result.relative_path,
                result.language or "-",
                str(len(result.matches)),
                f"{result.relevance:.0%}",
            )

        self.console.print(table)

        if show_context and results:
            # Show first few matches with context
            self.console.print("\n[bold]Top matches:[/bold]")
            for result in results[:3]:
                self.console.print(Panel(
                    f"[cyan]{result.file_path}[/cyan]",
                    title=f"{result.source_dir}/{result.relative_path}",
                    border_style="dim",
                ))
                for match in result.matches[:2]:
                    self.console.print(f"  Line {match['line_number']}: [yellow]{match['line'][:100]}[/yellow]")

        return results

    def extract_results(self, results: List[LocalSearchResult],
                       indices: List[int] = None) -> Dict[str, Any]:
        """Extract selected results to current project."""
        if indices:
            selected = [results[i-1] for i in indices if 0 < i <= len(results)]
        else:
            selected = results

        if not selected:
            return {"total": 0, "success": 0, "failed": 0}

        self.console.print(f"\n[bold]Extracting {len(selected)} files...[/bold]")

        summary = self.searcher.extract_multiple(selected)

        if summary["success"]:
            self.console.print(f"[green]Successfully extracted {summary['success']} files[/green]")
            for f in summary["extracted_files"][:5]:
                self.console.print(f"  [dim]{f}[/dim]")
            if len(summary["extracted_files"]) > 5:
                self.console.print(f"  [dim]... and {len(summary['extracted_files']) - 5} more[/dim]")

        if summary["failed"]:
            self.console.print(f"[red]Failed to extract {summary['failed']} files[/red]")
            for e in summary["errors"][:3]:
                self.console.print(f"  [dim]{e}[/dim]")

        return summary


# Global instance
_local_search_manager: Optional[LocalSearchManager] = None


def get_local_search_manager() -> LocalSearchManager:
    """Get or create the global local search manager."""
    global _local_search_manager
    if _local_search_manager is None:
        _local_search_manager = LocalSearchManager()
    return _local_search_manager


async def local_search(query: str, languages: List[str] = None,
                       extract: bool = False) -> List[LocalSearchResult]:
    """
    Convenience function for local code search.

    Args:
        query: Search query
        languages: Filter by languages
        extract: Whether to extract matches to current folder

    Returns:
        List of search results
    """
    manager = get_local_search_manager()
    results = manager.search_and_display(query, languages)

    if extract and results:
        manager.extract_results(results)

    return results

"""
Core Agent Engine for Coding Agent
Implements the fix → test → optimize → journal → push → repeat loop.
"""
import asyncio
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
import difflib

from config import config
from ai_providers import provider_manager, AIResponse
from local_ai import local_ai_manager


class AgentState(Enum):
    """States of the agent engine."""
    IDLE = "idle"
    ANALYZING = "analyzing"
    FIXING = "fixing"
    TESTING = "testing"
    OPTIMIZING = "optimizing"
    JOURNALING = "journaling"
    PUSHING = "pushing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CodeIssue:
    """Represents a code issue found during analysis."""
    file_path: str
    line_number: int
    issue_type: str  # error, warning, style, performance
    message: str
    code_snippet: str = ""
    suggested_fix: str = ""


@dataclass
class TestResult:
    """Result of running tests."""
    success: bool
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    output: str = ""
    duration_seconds: float = 0
    failed_tests: List[str] = field(default_factory=list)


@dataclass
class BuildResult:
    """Result of running a build/compile."""
    success: bool
    output: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0


@dataclass
class OptimizationResult:
    """Result of code optimization."""
    original_code: str
    optimized_code: str
    improvements: List[str]
    file_path: str


@dataclass
class AgentIteration:
    """Represents one iteration of the agent loop."""
    iteration_number: int
    state: AgentState
    issues_found: List[CodeIssue] = field(default_factory=list)
    fixes_applied: List[Dict[str, Any]] = field(default_factory=list)
    test_result: Optional[TestResult] = None
    build_result: Optional[BuildResult] = None
    optimizations: List[OptimizationResult] = field(default_factory=list)
    duration_seconds: float = 0
    ai_provider_used: str = ""
    tokens_used: int = 0


class CodeAnalyzer:
    """Analyzes code for issues using multiple methods."""

    def __init__(self):
        self.lint_patterns = {
            "python": ["ruff", "check", "--format=json"],
            "javascript": ["eslint", "--format=json"],
            "typescript": ["eslint", "--format=json"],
        }

    async def analyze_file(self, file_path: str) -> List[CodeIssue]:
        """Analyze a single file for issues."""
        issues = []
        path = Path(file_path)

        if not path.exists():
            return issues

        # Determine file type
        ext = path.suffix.lower()
        lang = self._get_language(ext)

        # Run linter
        lint_issues = await self._run_linter(file_path, lang)
        issues.extend(lint_issues)

        # Run syntax check
        syntax_issues = await self._check_syntax(file_path, lang)
        issues.extend(syntax_issues)

        return issues

    async def analyze_directory(self, directory: str,
                                 extensions: Optional[List[str]] = None) -> List[CodeIssue]:
        """Analyze all files in a directory."""
        issues = []
        path = Path(directory)

        if not path.exists():
            return issues

        exts = extensions or [".py", ".js", ".ts", ".jsx", ".tsx"]

        for file_path in path.rglob("*"):
            if file_path.suffix in exts and file_path.is_file():
                file_issues = await self.analyze_file(str(file_path))
                issues.extend(file_issues)

        return issues

    def _get_language(self, extension: str) -> str:
        """Get language from file extension."""
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
        }
        return mapping.get(extension, "unknown")

    async def _run_linter(self, file_path: str, language: str) -> List[CodeIssue]:
        """Run language-specific linter."""
        issues = []

        if language == "python":
            try:
                result = subprocess.run(
                    ["ruff", "check", "--output-format=json", file_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.stdout:
                    import json
                    lint_results = json.loads(result.stdout)
                    for item in lint_results:
                        issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=item.get("location", {}).get("row", 0),
                            issue_type="warning",
                            message=f"[{item.get('code')}] {item.get('message')}",
                            code_snippet=item.get("fix", {}).get("edits", [{}])[0].get("content", "")
                        ))
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                pass

        return issues

    async def _check_syntax(self, file_path: str, language: str) -> List[CodeIssue]:
        """Check for syntax errors."""
        issues = []

        if language == "python":
            try:
                result = subprocess.run(
                    ["python", "-m", "py_compile", file_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    # Parse error message
                    error_msg = result.stderr or result.stdout
                    line_match = re.search(r'line (\d+)', error_msg)
                    line_num = int(line_match.group(1)) if line_match else 0

                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type="error",
                        message=error_msg.strip()
                    ))
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        return issues


class TestRunner:
    """Runs tests and captures results."""

    def __init__(self):
        self.test_commands = {
            "pytest": ["pytest", "-v", "--tb=short"],
            "unittest": ["python", "-m", "unittest", "discover"],
            "jest": ["npx", "jest"],
            "mocha": ["npx", "mocha"],
            "go": ["go", "test", "./..."],
        }

    async def run_tests(self, test_command: Optional[str] = None,
                         working_dir: Optional[str] = None,
                         timeout: int = 300) -> TestResult:
        """Run tests and return results."""
        start_time = time.time()
        cmd = test_command or config.agent.test_command

        # Parse command
        if isinstance(cmd, str):
            cmd_parts = cmd.split()
        else:
            cmd_parts = cmd

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                cwd=working_dir or os.getcwd(),
                timeout=timeout
            )

            duration = time.time() - start_time

            # Parse test results
            return self._parse_test_output(
                result.stdout + result.stderr,
                result.returncode == 0,
                duration
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                output="Test execution timed out",
                duration_seconds=timeout
            )
        except FileNotFoundError as e:
            return TestResult(
                success=False,
                output=f"Test command not found: {e}",
                duration_seconds=0
            )

    def _parse_test_output(self, output: str, success: bool,
                           duration: float) -> TestResult:
        """Parse test output to extract metrics."""
        result = TestResult(
            success=success,
            output=output,
            duration_seconds=duration
        )

        # Try to parse pytest output
        pytest_match = re.search(
            r'(\d+) passed.*?(\d+)? failed.*?(\d+)? error',
            output, re.IGNORECASE
        )
        if pytest_match:
            result.passed = int(pytest_match.group(1) or 0)
            result.failed = int(pytest_match.group(2) or 0)
            result.errors = int(pytest_match.group(3) or 0)
            result.total_tests = result.passed + result.failed + result.errors

        # Extract failed test names
        failed_tests = re.findall(r'FAILED\s+(\S+)', output)
        result.failed_tests = failed_tests

        return result


class BuildRunner:
    """Runs build/compile commands."""

    async def run_build(self, build_command: Optional[str] = None,
                         working_dir: Optional[str] = None,
                         timeout: int = 120) -> BuildResult:
        """Run build command and return results."""
        start_time = time.time()
        cmd = build_command or config.agent.build_command

        if isinstance(cmd, str):
            cmd_parts = cmd.split()
        else:
            cmd_parts = cmd

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                cwd=working_dir or os.getcwd(),
                timeout=timeout
            )

            duration = time.time() - start_time
            output = result.stdout + result.stderr

            # Parse errors and warnings
            errors = re.findall(r'(?:error|Error|ERROR)[:\s](.+)', output)
            warnings = re.findall(r'(?:warning|Warning|WARNING)[:\s](.+)', output)

            return BuildResult(
                success=result.returncode == 0,
                output=output,
                errors=errors,
                warnings=warnings,
                duration_seconds=duration
            )

        except subprocess.TimeoutExpired:
            return BuildResult(
                success=False,
                output="Build timed out",
                duration_seconds=timeout
            )
        except FileNotFoundError as e:
            return BuildResult(
                success=False,
                output=f"Build command not found: {e}",
                duration_seconds=0
            )


class CodeFixer:
    """Fixes code issues using AI or rule-based approaches."""

    def __init__(self):
        self.use_ai = True

    async def fix_issue(self, issue: CodeIssue,
                         ai_provider: Optional[str] = None) -> Optional[str]:
        """
        Fix a code issue.

        Args:
            issue: The code issue to fix
            ai_provider: Optional specific AI provider to use

        Returns:
            Fixed code or None if fix failed
        """
        # Read the original file
        try:
            with open(issue.file_path, 'r') as f:
                original_code = f.read()
        except:
            return None

        if self.use_ai and (provider_manager.get_available_providers() or
                            local_ai_manager.backend):
            return await self._fix_with_ai(original_code, issue, ai_provider)
        else:
            return await self._fix_with_rules(original_code, issue)

    async def _fix_with_ai(self, code: str, issue: CodeIssue,
                            ai_provider: Optional[str] = None) -> Optional[str]:
        """Fix issue using AI."""
        try:
            # Try cloud AI first
            if ai_provider and ai_provider in provider_manager.providers:
                provider = provider_manager.get_provider(ai_provider)
                response = await provider.generate_code_fix(
                    code=code,
                    error=issue.message,
                    context=f"File: {issue.file_path}, Line: {issue.line_number}"
                )
                return self._extract_code(response.content)

            # Try any available cloud provider
            if provider_manager.get_available_providers():
                response = await provider_manager.generate_with_fallback(
                    prompt=f"""Fix this code error:

ERROR: {issue.message}
FILE: {issue.file_path}
LINE: {issue.line_number}

CODE:
```
{code}
```

Return only the fixed code.""",
                    system_prompt="You are an expert programmer. Fix the code error. Return only the corrected code."
                )
                return self._extract_code(response.content)

            # Fall back to local AI
            if local_ai_manager.backend and local_ai_manager.backend.model_loaded:
                fixed = await local_ai_manager.generate_code_fix(
                    code=code,
                    error=issue.message,
                    context=f"File: {issue.file_path}"
                )
                return self._extract_code(fixed)

        except Exception as e:
            print(f"AI fix failed: {e}")

        return None

    async def _fix_with_rules(self, code: str, issue: CodeIssue) -> Optional[str]:
        """Fix issue using rule-based approach (no AI needed)."""
        lines = code.split('\n')

        # Simple rule-based fixes
        if "undefined variable" in issue.message.lower():
            # Try to find the variable usage and add a definition
            pass

        if "missing import" in issue.message.lower():
            # Try to add missing import
            import_match = re.search(r"'(\w+)'", issue.message)
            if import_match:
                module = import_match.group(1)
                lines.insert(0, f"import {module}")
                return '\n'.join(lines)

        if "indentation" in issue.message.lower():
            # Try to fix indentation
            if issue.line_number > 0 and issue.line_number <= len(lines):
                idx = issue.line_number - 1
                lines[idx] = "    " + lines[idx].lstrip()
                return '\n'.join(lines)

        return None

    def _extract_code(self, response: str) -> str:
        """Extract code from AI response."""
        # Try to find code block
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Return as-is if no code block found
        return response.strip()

    def apply_fix(self, file_path: str, fixed_code: str) -> bool:
        """Apply a fix to a file."""
        try:
            # Backup original
            with open(file_path, 'r') as f:
                original = f.read()

            backup_path = file_path + '.backup'
            with open(backup_path, 'w') as f:
                f.write(original)

            # Write fixed code
            with open(file_path, 'w') as f:
                f.write(fixed_code)

            return True
        except Exception as e:
            print(f"Failed to apply fix: {e}")
            return False

    def generate_diff(self, original: str, fixed: str) -> str:
        """Generate a diff between original and fixed code."""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile='original',
            tofile='fixed'
        )
        return ''.join(diff)


class CodeOptimizer:
    """Optimizes code for performance, readability, etc."""

    async def optimize_file(self, file_path: str,
                             optimization_type: str = "performance") -> OptimizationResult:
        """Optimize a file's code."""
        with open(file_path, 'r') as f:
            original_code = f.read()

        # Try AI optimization
        if provider_manager.get_available_providers():
            try:
                response = await provider_manager.generate_with_fallback(
                    prompt=f"""Optimize this code for {optimization_type}:

```
{original_code}
```

Return only the optimized code.""",
                    system_prompt=f"You are an expert at code optimization. Optimize for {optimization_type}."
                )

                optimized = self._extract_code(response.content)
                improvements = self._identify_improvements(original_code, optimized)

                return OptimizationResult(
                    original_code=original_code,
                    optimized_code=optimized,
                    improvements=improvements,
                    file_path=file_path
                )
            except:
                pass

        # Return no optimization if AI unavailable
        return OptimizationResult(
            original_code=original_code,
            optimized_code=original_code,
            improvements=[],
            file_path=file_path
        )

    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        code_match = re.search(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response.strip()

    def _identify_improvements(self, original: str, optimized: str) -> List[str]:
        """Identify what improvements were made."""
        improvements = []

        # Count various metrics
        orig_lines = len(original.split('\n'))
        opt_lines = len(optimized.split('\n'))

        if opt_lines < orig_lines:
            improvements.append(f"Reduced code by {orig_lines - opt_lines} lines")

        # Check for common optimizations
        if "list comprehension" in optimized and "for" in original:
            improvements.append("Converted loops to list comprehensions")

        if "generator" in optimized.lower():
            improvements.append("Used generators for memory efficiency")

        return improvements


class AgentEngine:
    """
    Main agent engine that orchestrates the fix → test → optimize → journal → push loop.
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.analyzer = CodeAnalyzer()
        self.test_runner = TestRunner()
        self.build_runner = BuildRunner()
        self.fixer = CodeFixer()
        self.optimizer = CodeOptimizer()

        self.state = AgentState.IDLE
        self.iterations: List[AgentIteration] = []
        self.progress_callback = progress_callback

        # Settings
        self.max_iterations = config.agent.max_iterations
        self.auto_commit = config.git.auto_commit
        self.auto_push = config.git.auto_push

    def _update_state(self, state: AgentState, message: str = ""):
        """Update agent state and notify callback."""
        self.state = state
        if self.progress_callback:
            self.progress_callback(state.value, message)

    async def run(self, target: str, mode: str = "fix",
                   ai_provider: Optional[str] = None) -> List[AgentIteration]:
        """
        Run the agent loop.

        Args:
            target: File or directory to process
            mode: Operation mode (fix, optimize, full)
            ai_provider: Preferred AI provider

        Returns:
            List of iterations performed
        """
        self.iterations = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            iter_start = time.time()

            self._update_state(AgentState.ANALYZING,
                              f"Starting iteration {iteration}")

            current_iter = AgentIteration(
                iteration_number=iteration,
                state=AgentState.ANALYZING
            )

            # Step 1: Analyze
            issues = await self._analyze(target)
            current_iter.issues_found = issues

            if not issues:
                self._update_state(AgentState.TESTING, "No issues found, running tests...")

                # Step 2: Test
                test_result = await self._test()
                current_iter.test_result = test_result

                if test_result.success:
                    # Step 3: Optimize (optional)
                    if mode in ["optimize", "full"]:
                        self._update_state(AgentState.OPTIMIZING, "Optimizing code...")
                        optimizations = await self._optimize(target)
                        current_iter.optimizations = optimizations

                    current_iter.state = AgentState.COMPLETED
                    current_iter.duration_seconds = time.time() - iter_start
                    self.iterations.append(current_iter)

                    self._update_state(AgentState.COMPLETED,
                                      f"Completed successfully in {iteration} iteration(s)")
                    break

                else:
                    # Tests failed - extract issues from test output
                    issues = self._extract_issues_from_tests(test_result)
                    current_iter.issues_found = issues

            # Step 4: Fix issues
            if issues:
                self._update_state(AgentState.FIXING,
                                  f"Fixing {len(issues)} issue(s)...")

                fixes = await self._fix(issues, ai_provider)
                current_iter.fixes_applied = fixes

            # Step 5: Re-test
            self._update_state(AgentState.TESTING, "Re-running tests...")
            test_result = await self._test()
            current_iter.test_result = test_result

            # Step 6: Build check
            build_result = await self._build()
            current_iter.build_result = build_result

            current_iter.duration_seconds = time.time() - iter_start
            current_iter.state = AgentState.COMPLETED if (
                test_result.success and build_result.success
            ) else AgentState.FIXING

            self.iterations.append(current_iter)

            # Check if we're done
            if test_result.success and build_result.success:
                self._update_state(AgentState.COMPLETED,
                                  f"All tests pass after {iteration} iteration(s)")
                break

        else:
            self._update_state(AgentState.FAILED,
                              f"Max iterations ({self.max_iterations}) reached")

        return self.iterations

    async def run_standalone(self, target: str) -> List[AgentIteration]:
        """
        Run in standalone mode (no AI assistance).
        Uses only rule-based fixes and standard tools.
        """
        self.fixer.use_ai = False
        return await self.run(target, mode="fix")

    async def _analyze(self, target: str) -> List[CodeIssue]:
        """Analyze target for issues."""
        path = Path(target)

        if path.is_file():
            return await self.analyzer.analyze_file(target)
        elif path.is_dir():
            return await self.analyzer.analyze_directory(target)
        else:
            return []

    async def _test(self) -> TestResult:
        """Run tests."""
        return await self.test_runner.run_tests(
            timeout=config.agent.test_timeout
        )

    async def _build(self) -> BuildResult:
        """Run build/compile."""
        return await self.build_runner.run_build(
            timeout=config.agent.build_timeout
        )

    async def _fix(self, issues: List[CodeIssue],
                    ai_provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fix issues."""
        fixes = []

        for issue in issues:
            fixed_code = await self.fixer.fix_issue(issue, ai_provider)

            if fixed_code:
                # Apply the fix
                success = self.fixer.apply_fix(issue.file_path, fixed_code)

                fixes.append({
                    "file": issue.file_path,
                    "issue": issue.message,
                    "success": success,
                    "fix_applied": success
                })

        return fixes

    async def _optimize(self, target: str) -> List[OptimizationResult]:
        """Optimize code."""
        optimizations = []
        path = Path(target)

        if path.is_file():
            result = await self.optimizer.optimize_file(target)
            if result.improvements:
                optimizations.append(result)
                # Apply optimization
                with open(target, 'w') as f:
                    f.write(result.optimized_code)

        return optimizations

    def _extract_issues_from_tests(self, test_result: TestResult) -> List[CodeIssue]:
        """Extract issues from test failures."""
        issues = []

        # Parse test output for failures
        for failed_test in test_result.failed_tests:
            # Try to extract file and line from test name
            match = re.search(r'(\S+\.py)(?:::(\w+))?', failed_test)
            if match:
                file_path = match.group(1)
                issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=0,
                    issue_type="error",
                    message=f"Test failed: {failed_test}",
                    code_snippet=test_result.output[-500:]  # Last 500 chars
                ))

        return issues

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all iterations."""
        total_issues = sum(len(i.issues_found) for i in self.iterations)
        total_fixes = sum(len(i.fixes_applied) for i in self.iterations)
        total_optimizations = sum(len(i.optimizations) for i in self.iterations)
        total_duration = sum(i.duration_seconds for i in self.iterations)

        final_success = (
            self.iterations[-1].test_result.success
            if self.iterations and self.iterations[-1].test_result
            else False
        )

        return {
            "iterations": len(self.iterations),
            "total_issues_found": total_issues,
            "total_fixes_applied": total_fixes,
            "total_optimizations": total_optimizations,
            "total_duration_seconds": total_duration,
            "final_success": final_success,
            "final_state": self.state.value
        }


# Convenience function for simple usage
async def run_agent(target: str, mode: str = "fix",
                     ai_provider: Optional[str] = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Run the coding agent on a target.

    Args:
        target: File or directory to process
        mode: Operation mode (fix, optimize, full)
        ai_provider: Preferred AI provider
        progress_callback: Callback for progress updates

    Returns:
        Summary of the agent run
    """
    engine = AgentEngine(progress_callback=progress_callback)
    await engine.run(target, mode, ai_provider)
    return engine.get_summary()

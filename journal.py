"""
Journaling System for Coding Agent
Tracks all activities, iterations, and provides audit trail.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import hashlib

from config import JOURNAL_DIR, LOGS_DIR


class AgentJournal:
    """
    Maintains a comprehensive journal of all agent activities.
    Tracks iterations, fixes, optimizations, and AI interactions.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self.session_start = datetime.now()
        self.journal_file = JOURNAL_DIR / f"agent_{self.session_id}.json"
        self.log_file = LOGS_DIR / f"agent_{self.session_id}.log"
        self.entries: List[Dict[str, Any]] = []
        self._init_journal()

    def _generate_session_id(self) -> str:
        """Generate unique session ID based on timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"{timestamp}_{hash_suffix}"

    def _init_journal(self):
        """Initialize journal file with session metadata."""
        metadata = {
            "session_id": self.session_id,
            "started_at": self.session_start.isoformat(),
            "version": "1.0",
            "entries": []
        }
        self._save_journal(metadata)
        self.log("INFO", f"Session started: {self.session_id}")

    def _save_journal(self, data: Dict[str, Any]):
        """Save journal data to file."""
        with open(self.journal_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _load_journal(self) -> Dict[str, Any]:
        """Load journal data from file."""
        if self.journal_file.exists():
            with open(self.journal_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"session_id": self.session_id, "entries": []}

    def log(self, level: str, message: str):
        """Write log entry to log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

    def add_entry(self, entry_type: str, data: Dict[str, Any]) -> str:
        """
        Add a new entry to the journal.

        Args:
            entry_type: Type of entry
            data: Entry data

        Returns:
            Entry ID
        """
        entry_id = hashlib.md5(
            f"{datetime.now().timestamp()}{entry_type}".encode()
        ).hexdigest()[:12]

        entry = {
            "id": entry_id,
            "type": entry_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        journal = self._load_journal()
        journal["entries"].append(entry)
        self._save_journal(journal)

        self.log("INFO", f"Journal: {entry_type} - {entry_id}")
        return entry_id

    # Iteration tracking
    def record_iteration_start(self, iteration_number: int,
                                target: str, mode: str) -> str:
        """Record the start of an iteration."""
        return self.add_entry("iteration_start", {
            "iteration": iteration_number,
            "target": target,
            "mode": mode
        })

    def record_iteration_end(self, iteration_number: int,
                              success: bool, duration: float,
                              summary: Dict[str, Any]) -> str:
        """Record the end of an iteration."""
        return self.add_entry("iteration_end", {
            "iteration": iteration_number,
            "success": success,
            "duration_seconds": duration,
            "summary": summary
        })

    # Analysis tracking
    def record_analysis(self, target: str, issues_found: List[Dict]) -> str:
        """Record code analysis results."""
        return self.add_entry("analysis", {
            "target": target,
            "issues_count": len(issues_found),
            "issues": issues_found
        })

    # Fix tracking
    def record_fix_attempt(self, file_path: str, issue: str,
                           ai_provider: str, success: bool,
                           original_code: str = "",
                           fixed_code: str = "") -> str:
        """Record a fix attempt."""
        return self.add_entry("fix_attempt", {
            "file": file_path,
            "issue": issue,
            "provider": ai_provider,
            "success": success,
            "code_diff_lines": len(fixed_code.split('\n')) - len(original_code.split('\n'))
        })

    # Test tracking
    def record_test_run(self, command: str, result: Dict[str, Any]) -> str:
        """Record test execution."""
        return self.add_entry("test_run", {
            "command": command,
            "success": result.get("success", False),
            "passed": result.get("passed", 0),
            "failed": result.get("failed", 0),
            "duration": result.get("duration_seconds", 0)
        })

    # Build tracking
    def record_build(self, command: str, success: bool,
                      errors: List[str], warnings: List[str]) -> str:
        """Record build execution."""
        return self.add_entry("build", {
            "command": command,
            "success": success,
            "errors_count": len(errors),
            "warnings_count": len(warnings),
            "errors": errors[:10],  # Limit to first 10
            "warnings": warnings[:10]
        })

    # Optimization tracking
    def record_optimization(self, file_path: str,
                             improvements: List[str],
                             metrics_before: Dict[str, Any] = None,
                             metrics_after: Dict[str, Any] = None) -> str:
        """Record code optimization."""
        return self.add_entry("optimization", {
            "file": file_path,
            "improvements": improvements,
            "metrics_before": metrics_before or {},
            "metrics_after": metrics_after or {}
        })

    # AI interaction tracking
    def record_ai_interaction(self, provider: str, model: str,
                               prompt_type: str, tokens_used: int,
                               latency_ms: float, success: bool) -> str:
        """Record AI provider interaction."""
        return self.add_entry("ai_interaction", {
            "provider": provider,
            "model": model,
            "prompt_type": prompt_type,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "success": success
        })

    # Multi-agent tracking
    def record_agent_task(self, agent_id: str, task_id: str,
                           role: str, provider: str,
                           status: str, result: Dict[str, Any] = None) -> str:
        """Record multi-agent task execution."""
        return self.add_entry("agent_task", {
            "agent_id": agent_id,
            "task_id": task_id,
            "role": role,
            "provider": provider,
            "status": status,
            "result_summary": str(result)[:500] if result else None
        })

    def record_consensus(self, issue: str, agents_count: int,
                          agreed: bool, confidence: float) -> str:
        """Record consensus voting result."""
        return self.add_entry("consensus", {
            "issue": issue,
            "agents_participated": agents_count,
            "consensus_reached": agreed,
            "confidence": confidence
        })

    # Git tracking
    def record_git_action(self, action: str, details: Dict[str, Any]) -> str:
        """Record git operations."""
        return self.add_entry("git_action", {
            "action": action,
            **details
        })

    # Error tracking
    def record_error(self, context: str, error: str,
                      details: Optional[Dict] = None) -> str:
        """Record an error."""
        self.log("ERROR", f"{context}: {error}")
        return self.add_entry("error", {
            "context": context,
            "error": error,
            "details": details or {}
        })

    # Session management
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        journal = self._load_journal()
        entries = journal.get("entries", [])

        iterations = [e for e in entries if e["type"] == "iteration_end"]
        fixes = [e for e in entries if e["type"] == "fix_attempt"]
        tests = [e for e in entries if e["type"] == "test_run"]
        ai_calls = [e for e in entries if e["type"] == "ai_interaction"]
        errors = [e for e in entries if e["type"] == "error"]

        successful_fixes = [f for f in fixes if f["data"].get("success")]
        successful_tests = [t for t in tests if t["data"].get("success")]

        total_tokens = sum(
            e["data"].get("tokens_used", 0) for e in ai_calls
        )
        total_latency = sum(
            e["data"].get("latency_ms", 0) for e in ai_calls
        )

        # Calculate final status
        final_success = (
            iterations[-1]["data"]["success"]
            if iterations else False
        )

        return {
            "session_id": self.session_id,
            "duration": str(datetime.now() - self.session_start),
            "total_iterations": len(iterations),
            "total_fixes_attempted": len(fixes),
            "successful_fixes": len(successful_fixes),
            "total_test_runs": len(tests),
            "successful_test_runs": len(successful_tests),
            "total_ai_calls": len(ai_calls),
            "total_tokens_used": total_tokens,
            "total_ai_latency_ms": total_latency,
            "errors_count": len(errors),
            "final_success": final_success,
            "journal_file": str(self.journal_file),
            "log_file": str(self.log_file)
        }

    def close_session(self) -> Dict[str, Any]:
        """Close the session and finalize journal."""
        summary = self.get_session_summary()
        self.add_entry("session_end", summary)
        self.log("INFO", f"Session ended: {self.session_id}")
        return summary

    def export_report(self, format: str = "md") -> str:
        """
        Export session as a readable report.

        Args:
            format: Output format (md, txt, json)

        Returns:
            Path to exported file
        """
        summary = self.get_session_summary()
        journal = self._load_journal()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{self.session_id}_{timestamp}.{format}"
        filepath = JOURNAL_DIR / filename

        if format == "md":
            content = self._generate_markdown_report(summary, journal)
        elif format == "txt":
            content = self._generate_text_report(summary, journal)
        else:
            content = json.dumps({
                "summary": summary,
                "journal": journal
            }, indent=2, default=str)

        with open(filepath, 'w') as f:
            f.write(content)

        return str(filepath)

    def _generate_markdown_report(self, summary: Dict, journal: Dict) -> str:
        """Generate markdown report."""
        lines = [
            f"# Coding Agent Session Report",
            f"",
            f"**Session ID:** {summary['session_id']}",
            f"**Duration:** {summary['duration']}",
            f"**Final Status:** {'Success' if summary['final_success'] else 'Failed'}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Iterations | {summary['total_iterations']} |",
            f"| Fixes Attempted | {summary['total_fixes_attempted']} |",
            f"| Successful Fixes | {summary['successful_fixes']} |",
            f"| Test Runs | {summary['total_test_runs']} |",
            f"| Successful Tests | {summary['successful_test_runs']} |",
            f"| AI Calls | {summary['total_ai_calls']} |",
            f"| Tokens Used | {summary['total_tokens_used']} |",
            f"| Errors | {summary['errors_count']} |",
            f"",
            f"## Timeline",
            f""
        ]

        for entry in journal.get("entries", []):
            entry_type = entry["type"]
            timestamp = entry["timestamp"]
            data = entry["data"]

            if entry_type == "iteration_start":
                lines.append(f"### Iteration {data['iteration']}")
                lines.append(f"*Started at {timestamp}*")
                lines.append(f"")

            elif entry_type == "fix_attempt":
                status = "Fixed" if data["success"] else "Failed"
                lines.append(f"- [{status}] {data['file']}: {data['issue'][:50]}...")
                lines.append(f"  - Provider: {data['provider']}")

            elif entry_type == "test_run":
                status = "Passed" if data["success"] else "Failed"
                lines.append(f"- Tests: {status} ({data['passed']} passed, {data['failed']} failed)")

            elif entry_type == "error":
                lines.append(f"- **Error:** {data['context']}: {data['error'][:100]}")

        return "\n".join(lines)

    def _generate_text_report(self, summary: Dict, journal: Dict) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            "CODING AGENT SESSION REPORT",
            "=" * 60,
            "",
            f"Session ID: {summary['session_id']}",
            f"Duration: {summary['duration']}",
            f"Status: {'SUCCESS' if summary['final_success'] else 'FAILED'}",
            "",
            "-" * 60,
            "SUMMARY",
            "-" * 60,
            f"Iterations:       {summary['total_iterations']}",
            f"Fixes Attempted:  {summary['total_fixes_attempted']}",
            f"Successful Fixes: {summary['successful_fixes']}",
            f"Test Runs:        {summary['total_test_runs']}",
            f"AI Calls:         {summary['total_ai_calls']}",
            f"Tokens Used:      {summary['total_tokens_used']}",
            f"Errors:           {summary['errors_count']}",
            "",
        ]

        return "\n".join(lines)


# Global journal instance (created per session)
_current_journal: Optional[AgentJournal] = None


def get_journal() -> AgentJournal:
    """Get or create the current journal."""
    global _current_journal
    if _current_journal is None:
        _current_journal = AgentJournal()
    return _current_journal


def new_session() -> AgentJournal:
    """Start a new journal session."""
    global _current_journal
    if _current_journal:
        _current_journal.close_session()
    _current_journal = AgentJournal()
    return _current_journal

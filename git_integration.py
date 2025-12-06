"""
Git and GitHub Integration for Coding Agent
Handles commits, pushes, pull requests, and repository management.
"""
import asyncio
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from config import config


@dataclass
class GitStatus:
    """Current git repository status."""
    branch: str
    is_clean: bool
    staged_files: List[str]
    unstaged_files: List[str]
    untracked_files: List[str]
    ahead: int = 0
    behind: int = 0
    has_remote: bool = True


@dataclass
class CommitInfo:
    """Information about a commit."""
    hash: str
    short_hash: str
    message: str
    author: str
    date: str
    files_changed: int = 0


@dataclass
class GitHubRepo:
    """GitHub repository information."""
    owner: str
    name: str
    url: str
    ssh_url: str
    default_branch: str
    private: bool = False


class GitManager:
    """
    Manages git operations for the coding agent.
    Handles commits, pushes, and repository state.
    """

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = working_dir or os.getcwd()
        self._validate_git_repo()

    def _validate_git_repo(self) -> bool:
        """Check if current directory is a git repository."""
        git_dir = Path(self.working_dir) / ".git"
        if not git_dir.exists():
            # Try to find git root
            result = self._run_git(["rev-parse", "--git-dir"])
            if result.returncode != 0:
                return False
        return True

    def _run_git(self, args: List[str], check: bool = False) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + args
        return subprocess.run(
            cmd,
            cwd=self.working_dir,
            capture_output=True,
            text=True,
            check=check
        )

    async def _run_git_async(self, args: List[str]) -> Tuple[int, str, str]:
        """Run a git command asynchronously."""
        cmd = ["git"] + args
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    def get_status(self) -> GitStatus:
        """Get current git status."""
        # Get current branch
        result = self._run_git(["branch", "--show-current"])
        branch = result.stdout.strip() or "HEAD"

        # Get status
        result = self._run_git(["status", "--porcelain"])
        lines = result.stdout.strip().split('\n') if result.stdout.strip() else []

        staged = []
        unstaged = []
        untracked = []

        for line in lines:
            if len(line) < 3:
                continue
            status = line[:2]
            file_path = line[3:]

            if status[0] in 'MADRC':
                staged.append(file_path)
            if status[1] in 'MADRC':
                unstaged.append(file_path)
            if status == '??':
                untracked.append(file_path)

        # Check ahead/behind
        ahead = behind = 0
        result = self._run_git(["rev-list", "--left-right", "--count", f"{branch}...origin/{branch}"])
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])

        return GitStatus(
            branch=branch,
            is_clean=len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0,
            staged_files=staged,
            unstaged_files=unstaged,
            untracked_files=untracked,
            ahead=ahead,
            behind=behind,
            has_remote=self._has_remote()
        )

    def _has_remote(self) -> bool:
        """Check if repo has a remote."""
        result = self._run_git(["remote"])
        return bool(result.stdout.strip())

    def get_recent_commits(self, count: int = 5) -> List[CommitInfo]:
        """Get recent commits."""
        result = self._run_git([
            "log", f"-{count}",
            "--format=%H|%h|%s|%an|%ai"
        ])

        commits = []
        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 5:
                    commits.append(CommitInfo(
                        hash=parts[0],
                        short_hash=parts[1],
                        message=parts[2],
                        author=parts[3],
                        date=parts[4]
                    ))

        return commits

    def get_diff(self, staged: bool = False) -> str:
        """Get current diff."""
        args = ["diff"]
        if staged:
            args.append("--staged")
        result = self._run_git(args)
        return result.stdout

    def stage_files(self, files: List[str]) -> bool:
        """Stage files for commit."""
        if not files:
            return True
        result = self._run_git(["add"] + files)
        return result.returncode == 0

    def stage_all(self) -> bool:
        """Stage all changes."""
        result = self._run_git(["add", "-A"])
        return result.returncode == 0

    def commit(self, message: str, amend: bool = False) -> Tuple[bool, str]:
        """
        Create a commit.

        Args:
            message: Commit message
            amend: Whether to amend the previous commit

        Returns:
            Tuple of (success, commit hash or error message)
        """
        args = ["commit", "-m", message]
        if amend:
            args.append("--amend")

        result = self._run_git(args)

        if result.returncode == 0:
            # Get the commit hash
            hash_result = self._run_git(["rev-parse", "HEAD"])
            return True, hash_result.stdout.strip()[:8]
        else:
            return False, result.stderr

    def create_agent_commit(self, action: str, description: str,
                             files: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Create a commit with agent-style message.

        Args:
            action: Action type (fix, optimize, refactor, etc.)
            description: Description of changes
            files: Specific files to commit (None = all changes)

        Returns:
            Tuple of (success, commit hash or error message)
        """
        # Stage files
        if files:
            self.stage_files(files)
        else:
            self.stage_all()

        # Check if there's anything to commit
        status = self.get_status()
        if not status.staged_files:
            return False, "Nothing to commit"

        # Generate commit message from template
        message = config.git.commit_message_template.format(
            action=action,
            description=description
        )

        return self.commit(message)

    def push(self, force: bool = False, set_upstream: bool = False) -> Tuple[bool, str]:
        """
        Push commits to remote.

        Args:
            force: Force push (use with caution)
            set_upstream: Set upstream tracking

        Returns:
            Tuple of (success, output or error message)
        """
        args = ["push"]

        if force:
            args.append("--force")

        if set_upstream:
            status = self.get_status()
            args.extend(["-u", "origin", status.branch])

        result = self._run_git(args)

        if result.returncode == 0:
            return True, result.stdout or "Push successful"
        else:
            return False, result.stderr

    def pull(self, rebase: bool = False) -> Tuple[bool, str]:
        """Pull changes from remote."""
        args = ["pull"]
        if rebase:
            args.append("--rebase")

        result = self._run_git(args)
        return result.returncode == 0, result.stdout or result.stderr

    def create_branch(self, branch_name: str, checkout: bool = True) -> bool:
        """Create a new branch."""
        result = self._run_git(["checkout" if checkout else "branch", "-b", branch_name])
        return result.returncode == 0

    def checkout(self, branch: str) -> bool:
        """Checkout a branch."""
        result = self._run_git(["checkout", branch])
        return result.returncode == 0

    def stash(self, message: Optional[str] = None) -> bool:
        """Stash current changes."""
        args = ["stash"]
        if message:
            args.extend(["push", "-m", message])
        result = self._run_git(args)
        return result.returncode == 0

    def stash_pop(self) -> bool:
        """Pop stashed changes."""
        result = self._run_git(["stash", "pop"])
        return result.returncode == 0


class GitHubManager:
    """
    Manages GitHub operations using gh CLI.
    Handles repository creation, pull requests, and issues.
    """

    def __init__(self):
        self._validate_gh_cli()

    def _validate_gh_cli(self) -> bool:
        """Check if gh CLI is installed and authenticated."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _run_gh(self, args: List[str], check: bool = False) -> subprocess.CompletedProcess:
        """Run a gh command."""
        cmd = ["gh"] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )

    async def _run_gh_async(self, args: List[str]) -> Tuple[int, str, str]:
        """Run a gh command asynchronously."""
        cmd = ["gh"] + args
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return process.returncode, stdout.decode(), stderr.decode()

    def is_authenticated(self) -> bool:
        """Check if authenticated with GitHub."""
        result = self._run_gh(["auth", "status"])
        return result.returncode == 0

    def get_current_user(self) -> Optional[str]:
        """Get current authenticated user."""
        result = self._run_gh(["api", "user", "-q", ".login"])
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    def create_repo(self, name: str, description: str = "",
                    private: bool = False, org: Optional[str] = None) -> Tuple[bool, str]:
        """
        Create a new GitHub repository.

        Args:
            name: Repository name
            description: Repository description
            private: Whether repo is private
            org: Organization name (None for personal repo)

        Returns:
            Tuple of (success, repo URL or error message)
        """
        args = ["repo", "create"]

        if org:
            args.append(f"{org}/{name}")
        else:
            args.append(name)

        args.extend(["--description", description or "Created by Coding Agent"])

        if private:
            args.append("--private")
        else:
            args.append("--public")

        args.append("--confirm")

        result = self._run_gh(args)

        if result.returncode == 0:
            # Extract URL from output
            url_match = re.search(r'https://github.com/\S+', result.stdout + result.stderr)
            url = url_match.group(0) if url_match else f"https://github.com/{org or self.get_current_user()}/{name}"
            return True, url
        else:
            return False, result.stderr

    def add_remote(self, url: str, name: str = "origin",
                   working_dir: Optional[str] = None) -> bool:
        """Add a remote to the local repository."""
        git = GitManager(working_dir)
        result = git._run_git(["remote", "add", name, url])
        return result.returncode == 0

    def set_remote_url(self, url: str, name: str = "origin",
                       working_dir: Optional[str] = None) -> bool:
        """Set the URL for an existing remote."""
        git = GitManager(working_dir)
        result = git._run_git(["remote", "set-url", name, url])
        return result.returncode == 0

    def create_pr(self, title: str, body: str,
                   base: str = "main", draft: bool = False) -> Tuple[bool, str]:
        """
        Create a pull request.

        Args:
            title: PR title
            body: PR body/description
            base: Base branch to merge into
            draft: Create as draft PR

        Returns:
            Tuple of (success, PR URL or error message)
        """
        args = ["pr", "create",
                "--title", title,
                "--body", body,
                "--base", base]

        if draft:
            args.append("--draft")

        result = self._run_gh(args)

        if result.returncode == 0:
            url_match = re.search(r'https://github.com/\S+', result.stdout)
            return True, url_match.group(0) if url_match else result.stdout.strip()
        else:
            return False, result.stderr

    def list_repos(self, user: Optional[str] = None,
                    limit: int = 30) -> List[Dict[str, Any]]:
        """List repositories for a user."""
        args = ["repo", "list"]
        if user:
            args.append(user)
        args.extend(["--limit", str(limit), "--json", "name,url,isPrivate,defaultBranchRef"])

        result = self._run_gh(args)

        if result.returncode == 0:
            import json
            try:
                return json.loads(result.stdout)
            except:
                return []
        return []

    def clone_repo(self, repo: str, directory: Optional[str] = None) -> Tuple[bool, str]:
        """Clone a repository."""
        args = ["repo", "clone", repo]
        if directory:
            args.append(directory)

        result = self._run_gh(args)
        return result.returncode == 0, result.stdout or result.stderr

    def fork_repo(self, repo: str) -> Tuple[bool, str]:
        """Fork a repository."""
        result = self._run_gh(["repo", "fork", repo, "--clone=false"])
        return result.returncode == 0, result.stdout or result.stderr


class GitHubAccountManager:
    """
    Manages multiple GitHub accounts as specified in config.
    Handles switching between accounts and creating repos on each.
    """

    def __init__(self):
        self.accounts = config.git.github_accounts
        self.gh = GitHubManager()

    def create_repos_on_accounts(self, repo_name: str, description: str = "",
                                   private: bool = False) -> Dict[str, Tuple[bool, str]]:
        """
        Create repository on all configured GitHub accounts.

        Args:
            repo_name: Name for the repository
            description: Repository description
            private: Whether repos should be private

        Returns:
            Dict mapping account to (success, url_or_error)
        """
        results = {}

        for account in self.accounts:
            # Try to create on the account (as org)
            success, result = self.gh.create_repo(
                name=repo_name,
                description=description,
                private=private,
                org=account if account != self.gh.get_current_user() else None
            )
            results[account] = (success, result)

        return results

    def setup_multiple_remotes(self, repo_name: str,
                                working_dir: Optional[str] = None) -> Dict[str, bool]:
        """
        Set up remotes for all configured accounts.

        Args:
            repo_name: Repository name
            working_dir: Working directory

        Returns:
            Dict mapping account to success status
        """
        git = GitManager(working_dir)
        results = {}

        for i, account in enumerate(self.accounts):
            remote_name = "origin" if i == 0 else account
            url = f"git@github.com:{account}/{repo_name}.git"

            if i == 0:
                # First account uses origin
                success = git._run_git(["remote", "add", "origin", url]).returncode == 0
            else:
                # Additional accounts get named remotes
                success = git._run_git(["remote", "add", remote_name, url]).returncode == 0

            results[account] = success

        return results

    async def push_to_all_accounts(self, working_dir: Optional[str] = None,
                                     branch: str = "main") -> Dict[str, Tuple[bool, str]]:
        """
        Push to all configured account remotes.

        Args:
            working_dir: Working directory
            branch: Branch to push

        Returns:
            Dict mapping account to (success, message)
        """
        git = GitManager(working_dir)
        results = {}

        for i, account in enumerate(self.accounts):
            remote_name = "origin" if i == 0 else account

            result = git._run_git(["push", "-u", remote_name, branch])
            results[account] = (result.returncode == 0, result.stdout or result.stderr)

        return results


# Convenience functions
def quick_commit(message: str, files: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Quick commit helper."""
    git = GitManager()
    if files:
        git.stage_files(files)
    else:
        git.stage_all()
    return git.commit(message)


def quick_push() -> Tuple[bool, str]:
    """Quick push helper."""
    git = GitManager()
    return git.push()


async def create_and_push_repos(repo_name: str, description: str = "",
                                  private: bool = False) -> Dict[str, Any]:
    """
    Create repos on all configured accounts and push code.

    Args:
        repo_name: Repository name
        description: Repository description
        private: Whether repos should be private

    Returns:
        Results of creation and push operations
    """
    account_mgr = GitHubAccountManager()
    git = GitManager()

    results = {
        "repo_creation": {},
        "remote_setup": {},
        "push": {}
    }

    # Create repos
    results["repo_creation"] = account_mgr.create_repos_on_accounts(
        repo_name, description, private
    )

    # Set up remotes
    results["remote_setup"] = account_mgr.setup_multiple_remotes(repo_name)

    # Push to all
    status = git.get_status()
    results["push"] = await account_mgr.push_to_all_accounts(branch=status.branch)

    return results

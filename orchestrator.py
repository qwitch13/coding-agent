"""
Multi-Agent Orchestrator for Coding Agent
Coordinates multiple AI agents working in parallel for faster optimization.
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Set
from collections import defaultdict
import hashlib
import json

from config import config
from ai_providers import provider_manager, AIResponse, BaseAIProvider
from local_ai import local_ai_manager
from agent_engine import (
    AgentEngine, AgentState, AgentIteration,
    CodeIssue, TestResult, BuildResult
)


class AgentRole(Enum):
    """Roles that agents can take in multi-agent collaboration."""
    ANALYZER = "analyzer"       # Finds issues in code
    FIXER = "fixer"            # Fixes code issues
    TESTER = "tester"          # Runs and analyzes tests
    OPTIMIZER = "optimizer"    # Optimizes code
    REVIEWER = "reviewer"      # Reviews fixes from other agents
    COORDINATOR = "coordinator" # Coordinates other agents


@dataclass
class AgentTask:
    """A task assigned to an agent."""
    task_id: str
    role: AgentRole
    target: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    assigned_agent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class Agent:
    """Represents a single agent in the swarm."""
    agent_id: str
    provider: str  # claude, chatgpt, gemini, perplexity, local, junie
    role: AgentRole
    status: str = "idle"  # idle, busy, failed
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tokens_used: int = 0
    total_latency_ms: float = 0


@dataclass
class ConsensusResult:
    """Result of consensus voting among agents."""
    agreed: bool
    winning_response: str
    votes: Dict[str, int]
    all_responses: List[Dict[str, Any]]
    confidence: float


class TaskQueue:
    """Priority queue for agent tasks."""

    def __init__(self):
        self.tasks: Dict[str, AgentTask] = {}
        self.pending: List[str] = []
        self.running: Set[str] = set()
        self.completed: List[str] = []
        self.failed: List[str] = []

    def add_task(self, task: AgentTask):
        """Add a task to the queue."""
        self.tasks[task.task_id] = task
        self.pending.append(task.task_id)
        # Sort by priority
        self.pending.sort(key=lambda tid: self.tasks[tid].priority, reverse=True)

    def get_next_task(self, exclude_deps: bool = True) -> Optional[AgentTask]:
        """Get the next available task."""
        for task_id in self.pending:
            task = self.tasks[task_id]

            # Check dependencies
            if exclude_deps and task.dependencies:
                deps_complete = all(
                    self.tasks.get(dep_id, AgentTask("", AgentRole.FIXER, "")).status == "completed"
                    for dep_id in task.dependencies
                )
                if not deps_complete:
                    continue

            return task

        return None

    def start_task(self, task_id: str, agent_id: str):
        """Mark a task as started."""
        if task_id in self.pending:
            self.pending.remove(task_id)
            self.running.add(task_id)
            self.tasks[task_id].status = "running"
            self.tasks[task_id].assigned_agent = agent_id
            self.tasks[task_id].started_at = time.time()

    def complete_task(self, task_id: str, result: Any):
        """Mark a task as completed."""
        if task_id in self.running:
            self.running.remove(task_id)
            self.completed.append(task_id)
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].result = result
            self.tasks[task_id].completed_at = time.time()

    def fail_task(self, task_id: str, error: str):
        """Mark a task as failed."""
        if task_id in self.running:
            self.running.remove(task_id)
            self.failed.append(task_id)
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].result = {"error": error}
            self.tasks[task_id].completed_at = time.time()

    def has_pending(self) -> bool:
        """Check if there are pending tasks."""
        return len(self.pending) > 0

    def all_complete(self) -> bool:
        """Check if all tasks are complete (or failed)."""
        return len(self.pending) == 0 and len(self.running) == 0


class AgentOrchestrator:
    """
    Orchestrates multiple AI agents working in parallel.

    Strategies:
    - parallel: Run multiple agents on different tasks simultaneously
    - consensus: Run multiple agents on same task and vote on best solution
    - pipeline: Chain agents in a pipeline (analyze → fix → review → optimize)
    - swarm: Dynamic task allocation based on agent availability
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        self.agents: Dict[str, Agent] = {}
        self.task_queue = TaskQueue()
        self.progress_callback = progress_callback
        self.strategy = "parallel"

        # Results storage
        self.iteration_results: List[Dict[str, Any]] = []

        # Initialize available agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize agents based on available providers."""
        agent_id = 0

        # Cloud providers
        for provider_name in provider_manager.get_available_providers():
            self.agents[f"agent_{agent_id}"] = Agent(
                agent_id=f"agent_{agent_id}",
                provider=provider_name,
                role=AgentRole.FIXER
            )
            agent_id += 1

        # Local AI
        if config.local_ai.enabled and local_ai_manager.backend:
            self.agents[f"agent_{agent_id}"] = Agent(
                agent_id=f"agent_{agent_id}",
                provider="local",
                role=AgentRole.FIXER
            )
            agent_id += 1

        # If no agents available, create a standalone agent
        if not self.agents:
            self.agents["agent_0"] = Agent(
                agent_id="agent_0",
                provider="standalone",
                role=AgentRole.FIXER
            )

    def _notify_progress(self, event: str, data: Dict[str, Any]):
        """Notify progress callback."""
        if self.progress_callback:
            self.progress_callback(event, data)

    def _generate_task_id(self, role: AgentRole, target: str) -> str:
        """Generate unique task ID."""
        content = f"{role.value}:{target}:{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def _execute_agent_task(self, agent: Agent, task: AgentTask) -> Any:
        """Execute a task with a specific agent."""
        agent.status = "busy"
        agent.current_task = task.task_id
        start_time = time.time()

        try:
            result = None

            if task.role == AgentRole.ANALYZER:
                result = await self._run_analysis(agent, task)
            elif task.role == AgentRole.FIXER:
                result = await self._run_fix(agent, task)
            elif task.role == AgentRole.TESTER:
                result = await self._run_test(agent, task)
            elif task.role == AgentRole.OPTIMIZER:
                result = await self._run_optimize(agent, task)
            elif task.role == AgentRole.REVIEWER:
                result = await self._run_review(agent, task)

            agent.tasks_completed += 1
            agent.total_latency_ms += (time.time() - start_time) * 1000

            return result

        except Exception as e:
            return {"error": str(e)}

        finally:
            agent.status = "idle"
            agent.current_task = None

    async def _run_analysis(self, agent: Agent, task: AgentTask) -> Dict[str, Any]:
        """Run code analysis with an agent."""
        target = task.target
        context = task.context

        # Use appropriate provider
        if agent.provider in provider_manager.providers:
            provider = provider_manager.get_provider(agent.provider)

            with open(target, 'r') as f:
                code = f.read()

            response = await provider.analyze_code(code, "review")
            return {
                "analysis": response.content,
                "provider": agent.provider,
                "tokens_used": response.tokens_used
            }

        return {"analysis": "", "provider": agent.provider}

    async def _run_fix(self, agent: Agent, task: AgentTask) -> Dict[str, Any]:
        """Run code fix with an agent."""
        issue = task.context.get("issue")
        code = task.context.get("code", "")

        if agent.provider in provider_manager.providers:
            provider = provider_manager.get_provider(agent.provider)
            response = await provider.generate_code_fix(
                code=code,
                error=issue.message if issue else "Fix any issues",
                context=task.target
            )

            return {
                "fixed_code": response.content,
                "provider": agent.provider,
                "tokens_used": response.tokens_used
            }

        elif agent.provider == "local" and local_ai_manager.backend:
            fixed = await local_ai_manager.generate_code_fix(
                code=code,
                error=issue.message if issue else "Fix any issues",
                context=task.target
            )
            return {
                "fixed_code": fixed,
                "provider": "local",
                "tokens_used": 0
            }

        return {"fixed_code": code, "provider": agent.provider}

    async def _run_test(self, agent: Agent, task: AgentTask) -> Dict[str, Any]:
        """Run tests (uses local test runner, agent analyzes results)."""
        from agent_engine import TestRunner
        runner = TestRunner()
        result = await runner.run_tests()

        if not result.success and agent.provider in provider_manager.providers:
            # Have AI analyze failures
            provider = provider_manager.get_provider(agent.provider)
            analysis = await provider.generate(
                f"Analyze these test failures and suggest fixes:\n{result.output}",
                system_prompt="You are a test expert. Analyze failures and suggest specific fixes."
            )
            return {
                "test_result": result,
                "analysis": analysis.content,
                "provider": agent.provider
            }

        return {"test_result": result, "provider": agent.provider}

    async def _run_optimize(self, agent: Agent, task: AgentTask) -> Dict[str, Any]:
        """Run code optimization with an agent."""
        target = task.target

        with open(target, 'r') as f:
            code = f.read()

        if agent.provider in provider_manager.providers:
            provider = provider_manager.get_provider(agent.provider)
            response = await provider.generate(
                f"Optimize this code for performance and readability:\n```\n{code}\n```",
                system_prompt="You are an optimization expert. Return only the optimized code."
            )

            return {
                "optimized_code": response.content,
                "provider": agent.provider,
                "tokens_used": response.tokens_used
            }

        return {"optimized_code": code, "provider": agent.provider}

    async def _run_review(self, agent: Agent, task: AgentTask) -> Dict[str, Any]:
        """Review code changes with an agent."""
        original = task.context.get("original", "")
        modified = task.context.get("modified", "")

        if agent.provider in provider_manager.providers:
            provider = provider_manager.get_provider(agent.provider)
            response = await provider.generate(
                f"""Review this code change:

ORIGINAL:
```
{original}
```

MODIFIED:
```
{modified}
```

Is the change correct? Are there any issues?""",
                system_prompt="You are a code reviewer. Evaluate if the change is correct and safe."
            )

            # Parse approval from response
            approved = any(word in response.content.lower()
                          for word in ["approved", "correct", "good", "lgtm", "looks good"])

            return {
                "review": response.content,
                "approved": approved,
                "provider": agent.provider,
                "tokens_used": response.tokens_used
            }

        return {"review": "", "approved": True, "provider": agent.provider}

    async def run_parallel(self, target: str,
                            max_agents: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run agents in parallel on different aspects of the target.

        Args:
            target: File or directory to process
            max_agents: Maximum number of agents to use

        Returns:
            List of results from all agents
        """
        num_agents = min(
            max_agents or config.agent.max_parallel_agents,
            len(self.agents)
        )

        # Create tasks for different roles
        tasks = [
            AgentTask(
                task_id=self._generate_task_id(AgentRole.ANALYZER, target),
                role=AgentRole.ANALYZER,
                target=target,
                priority=3
            ),
            AgentTask(
                task_id=self._generate_task_id(AgentRole.TESTER, target),
                role=AgentRole.TESTER,
                target=target,
                priority=2
            ),
        ]

        # Add tasks to queue
        for task in tasks:
            self.task_queue.add_task(task)

        results = []
        available_agents = list(self.agents.values())[:num_agents]

        # Run tasks in parallel
        async def run_agent(agent: Agent):
            while not self.task_queue.all_complete():
                task = self.task_queue.get_next_task()
                if task:
                    self.task_queue.start_task(task.task_id, agent.agent_id)
                    self._notify_progress("task_started", {
                        "task_id": task.task_id,
                        "agent": agent.agent_id,
                        "role": task.role.value
                    })

                    result = await self._execute_agent_task(agent, task)
                    self.task_queue.complete_task(task.task_id, result)

                    self._notify_progress("task_completed", {
                        "task_id": task.task_id,
                        "agent": agent.agent_id,
                        "success": "error" not in result
                    })

                    results.append({
                        "task_id": task.task_id,
                        "agent": agent.agent_id,
                        "role": task.role.value,
                        "result": result
                    })
                else:
                    await asyncio.sleep(0.1)

        await asyncio.gather(*[run_agent(agent) for agent in available_agents])

        return results

    async def run_consensus(self, code: str, issue: CodeIssue,
                             min_agents: int = 2) -> ConsensusResult:
        """
        Run multiple agents on the same fix and reach consensus.

        Args:
            code: Code to fix
            issue: Issue to fix
            min_agents: Minimum number of agents required

        Returns:
            ConsensusResult with the winning fix
        """
        available_agents = [a for a in self.agents.values()
                           if a.provider != "standalone"]

        if len(available_agents) < min_agents:
            # Not enough agents for consensus
            if available_agents:
                task = AgentTask(
                    task_id=self._generate_task_id(AgentRole.FIXER, issue.file_path),
                    role=AgentRole.FIXER,
                    target=issue.file_path,
                    context={"issue": issue, "code": code}
                )
                result = await self._execute_agent_task(available_agents[0], task)
                return ConsensusResult(
                    agreed=True,
                    winning_response=result.get("fixed_code", code),
                    votes={available_agents[0].provider: 1},
                    all_responses=[result],
                    confidence=1.0
                )

            return ConsensusResult(
                agreed=False,
                winning_response=code,
                votes={},
                all_responses=[],
                confidence=0.0
            )

        # Run all agents in parallel
        tasks = []
        for agent in available_agents[:min_agents]:
            task = AgentTask(
                task_id=self._generate_task_id(AgentRole.FIXER, issue.file_path),
                role=AgentRole.FIXER,
                target=issue.file_path,
                context={"issue": issue, "code": code}
            )
            tasks.append(self._execute_agent_task(agent, task))

        results = await asyncio.gather(*tasks)

        # Vote on responses
        # Simple voting: count similar responses
        responses = [r.get("fixed_code", "") for r in results]
        votes = defaultdict(int)

        for resp in responses:
            # Normalize response for comparison
            normalized = resp.strip()
            votes[normalized] += 1

        # Find winner
        winning_response = max(votes.keys(), key=lambda k: votes[k])
        max_votes = votes[winning_response]
        total_votes = len(responses)
        confidence = max_votes / total_votes if total_votes > 0 else 0

        return ConsensusResult(
            agreed=confidence >= 0.5,
            winning_response=winning_response,
            votes={k[:50]: v for k, v in votes.items()},  # Truncate keys
            all_responses=results,
            confidence=confidence
        )

    async def run_pipeline(self, target: str) -> Dict[str, Any]:
        """
        Run agents in a pipeline: analyze → fix → review → optimize.

        Args:
            target: File to process

        Returns:
            Final result after all pipeline stages
        """
        available_agents = list(self.agents.values())
        if not available_agents:
            return {"error": "No agents available"}

        results = {"stages": []}

        # Stage 1: Analyze
        self._notify_progress("pipeline_stage", {"stage": "analyze"})
        analyzer = available_agents[0]
        analyze_task = AgentTask(
            task_id=self._generate_task_id(AgentRole.ANALYZER, target),
            role=AgentRole.ANALYZER,
            target=target
        )
        analyze_result = await self._execute_agent_task(analyzer, analyze_task)
        results["stages"].append({"analyze": analyze_result})

        # Read current code
        with open(target, 'r') as f:
            original_code = f.read()

        # Stage 2: Fix (if issues found)
        self._notify_progress("pipeline_stage", {"stage": "fix"})
        fixer = available_agents[min(1, len(available_agents) - 1)]
        fix_task = AgentTask(
            task_id=self._generate_task_id(AgentRole.FIXER, target),
            role=AgentRole.FIXER,
            target=target,
            context={
                "code": original_code,
                "issue": CodeIssue(
                    file_path=target,
                    line_number=0,
                    issue_type="analysis",
                    message=analyze_result.get("analysis", "Improve this code")
                )
            }
        )
        fix_result = await self._execute_agent_task(fixer, fix_task)
        results["stages"].append({"fix": fix_result})

        fixed_code = fix_result.get("fixed_code", original_code)

        # Stage 3: Review
        self._notify_progress("pipeline_stage", {"stage": "review"})
        reviewer = available_agents[min(2, len(available_agents) - 1)]
        review_task = AgentTask(
            task_id=self._generate_task_id(AgentRole.REVIEWER, target),
            role=AgentRole.REVIEWER,
            target=target,
            context={
                "original": original_code,
                "modified": fixed_code
            }
        )
        review_result = await self._execute_agent_task(reviewer, review_task)
        results["stages"].append({"review": review_result})

        # Stage 4: Optimize (if review approved)
        if review_result.get("approved", True):
            self._notify_progress("pipeline_stage", {"stage": "optimize"})
            optimizer = available_agents[min(3, len(available_agents) - 1)]
            optimize_task = AgentTask(
                task_id=self._generate_task_id(AgentRole.OPTIMIZER, target),
                role=AgentRole.OPTIMIZER,
                target=target
            )

            # Temporarily write fixed code for optimization
            with open(target, 'w') as f:
                f.write(fixed_code)

            optimize_result = await self._execute_agent_task(optimizer, optimize_task)
            results["stages"].append({"optimize": optimize_result})

            final_code = optimize_result.get("optimized_code", fixed_code)
        else:
            final_code = original_code  # Revert if not approved

        # Write final code
        with open(target, 'w') as f:
            f.write(final_code)

        results["final_code"] = final_code
        results["approved"] = review_result.get("approved", False)

        return results

    async def run_swarm(self, target: str,
                         max_iterations: int = 10) -> List[Dict[str, Any]]:
        """
        Run a swarm of agents with dynamic task allocation.
        Agents pick up tasks as they become available.

        Args:
            target: File or directory to process
            max_iterations: Maximum optimization iterations

        Returns:
            List of all task results
        """
        from agent_engine import CodeAnalyzer, TestRunner

        analyzer = CodeAnalyzer()
        test_runner = TestRunner()

        all_results = []
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self._notify_progress("swarm_iteration", {"iteration": iteration})

            # Analyze current state
            issues = await analyzer.analyze_file(target) if Path(target).is_file() else \
                    await analyzer.analyze_directory(target)

            # Run tests
            test_result = await test_runner.run_tests()

            # If everything passes, we're done
            if not issues and test_result.success:
                self._notify_progress("swarm_complete", {
                    "iterations": iteration,
                    "success": True
                })
                break

            # Create fix tasks for each issue
            for issue in issues[:config.agent.max_parallel_agents]:
                with open(issue.file_path, 'r') as f:
                    code = f.read()

                task = AgentTask(
                    task_id=self._generate_task_id(AgentRole.FIXER, issue.file_path),
                    role=AgentRole.FIXER,
                    target=issue.file_path,
                    context={"issue": issue, "code": code},
                    priority=1 if issue.issue_type == "error" else 0
                )
                self.task_queue.add_task(task)

            # Run agents on available tasks
            results = await self.run_parallel(target)
            all_results.extend(results)

            # Apply successful fixes
            for result in results:
                if result["role"] == "fixer" and "error" not in result.get("result", {}):
                    fixed_code = result["result"].get("fixed_code")
                    if fixed_code:
                        task = self.task_queue.tasks.get(result["task_id"])
                        if task:
                            with open(task.target, 'w') as f:
                                f.write(fixed_code)

        return all_results

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "agents": {
                agent_id: {
                    "provider": agent.provider,
                    "role": agent.role.value,
                    "status": agent.status,
                    "tasks_completed": agent.tasks_completed,
                    "tokens_used": agent.tokens_used
                }
                for agent_id, agent in self.agents.items()
            },
            "tasks": {
                "pending": len(self.task_queue.pending),
                "running": len(self.task_queue.running),
                "completed": len(self.task_queue.completed),
                "failed": len(self.task_queue.failed)
            },
            "strategy": self.strategy
        }


# Import Path for file operations
from pathlib import Path


# Convenience function
async def run_multi_agent(target: str, strategy: str = "parallel",
                           progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Run multi-agent optimization on a target.

    Args:
        target: File or directory to process
        strategy: Strategy to use (parallel, consensus, pipeline, swarm)
        progress_callback: Callback for progress updates

    Returns:
        Results from the multi-agent run
    """
    orchestrator = AgentOrchestrator(progress_callback=progress_callback)

    if strategy == "parallel":
        results = await orchestrator.run_parallel(target)
    elif strategy == "pipeline":
        results = await orchestrator.run_pipeline(target)
    elif strategy == "swarm":
        results = await orchestrator.run_swarm(target)
    else:
        results = await orchestrator.run_parallel(target)

    return {
        "strategy": strategy,
        "results": results,
        "status": orchestrator.get_status()
    }

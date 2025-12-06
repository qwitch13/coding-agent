# Coding Agent

Autonomous code optimization system that iteratively fixes, tests, optimizes, journals, and pushes code until it works flawlessly.

## Features

- **Iterative Optimization Loop**: Fix -> Test -> Optimize -> Journal -> Push -> Repeat
- **Multiple AI Providers**: Claude, ChatGPT, Gemini, Perplexity, and Junie (JetBrains)
- **Local AI Support**: Optimized for Apple Silicon M3/M4 chips using MLX, llama.cpp, or Ollama
- **Multi-Agent Orchestration**: Run multiple AI agents in parallel for faster optimization
- **Standalone Mode**: Works without AI using rule-based fixes
- **User Interaction**: Asks for input, resources, and clarifications when needed
- **Comprehensive Journaling**: Tracks all activities, iterations, and decisions
- **Git Integration**: Automatic commits and pushes to multiple GitHub accounts

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys (optional - works without AI too)
export ANTHROPIC_API_KEY=your_claude_key
export OPENAI_API_KEY=your_openai_key
export GOOGLE_API_KEY=your_gemini_key
export PERPLEXITY_API_KEY=your_perplexity_key

# Run interactive mode
python main.py

# Or run on a specific target
python main.py run src/
python main.py fix main.py
python main.py optimize utils.py
```

## Modes

| Mode | Description |
|------|-------------|
| `standalone` | No AI, rule-based fixes only |
| `single_ai` | Use one AI provider |
| `multi_agent` | Use multiple AI agents in parallel |

```bash
# Set mode
python main.py --mode multi_agent run src/
```

## Commands

| Command | Description |
|---------|-------------|
| `run <target>` | Run full optimization loop on file/directory |
| `fix <target>` | Fix issues only |
| `optimize <target>` | Optimize code |
| `test` | Run tests |
| `status` | Show current status |
| `journal` | Show session journal |
| `providers` | List available AI providers |
| `push` | Commit and push changes |
| `create-repos` | Create GitHub repos on nebulai13 and qwitch13 |

## Architecture

```
coding-agent/
├── main.py              # Entry point and CLI
├── config.py            # Configuration management
├── ai_providers.py      # Claude, ChatGPT, Gemini, Perplexity, Junie
├── local_ai.py          # Local AI (MLX, llama.cpp, Ollama)
├── agent_engine.py      # Core fix/test/optimize loop
├── orchestrator.py      # Multi-agent coordination
├── journal.py           # Activity logging
├── git_integration.py   # Git and GitHub operations
├── terminal_ui.py       # Rich terminal interface
├── user_interaction.py  # User prompts and input
└── requirements.txt     # Dependencies
```

## Local AI (Apple Silicon M3/M4)

The agent supports local AI inference optimized for Apple Silicon:

```bash
# Using Ollama (recommended for easy setup)
ollama pull codellama:7b-instruct
python main.py --mode single_ai run src/

# Using MLX (best performance on Apple Silicon)
pip install mlx mlx-lm
# Download a model to models/ directory
python main.py run src/
```

### Recommended Models

| Model | Size | Best For |
|-------|------|----------|
| CodeLlama 7B | 4.1 GB | General code fixes |
| DeepSeek Coder 6.7B | 4.0 GB | Code generation |
| Qwen2.5 Coder 7B | 4.5 GB | Multi-language support |

## Multi-Agent Strategies

| Strategy | Description |
|----------|-------------|
| `parallel` | Run agents on different tasks simultaneously |
| `consensus` | Multiple agents vote on best solution |
| `pipeline` | Chain agents: analyze -> fix -> review -> optimize |
| `swarm` | Dynamic task allocation based on availability |

## User Interaction

The agent asks for input when needed:

- **API Keys**: Prompts for missing credentials
- **Clarifications**: Asks about ambiguous errors
- **Resource Requests**: Requests files or configs
- **Code Review**: Shows diffs before applying changes
- **Confirmations**: Asks before destructive actions

## Environment Variables

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Claude |
| `OPENAI_API_KEY` | ChatGPT |
| `GOOGLE_API_KEY` | Gemini |
| `PERPLEXITY_API_KEY` | Perplexity |

## GitHub Integration

Create repos on multiple accounts:

```bash
# Create repos on nebulai13 and qwitch13
python main.py create-repos --name my-project

# Push to all configured remotes
python main.py push
```

## Configuration

Edit `agent_config.json` or use environment variables:

```json
{
  "agent": {
    "mode": "multi_agent",
    "max_iterations": 10,
    "test_command": "pytest",
    "build_command": "python -m py_compile"
  },
  "local_ai": {
    "enabled": true,
    "backend": "mlx",
    "model_name": "codellama-7b"
  },
  "git": {
    "auto_commit": false,
    "auto_push": false,
    "github_accounts": ["nebulai13", "qwitch13"]
  }
}
```

## Requirements

- Python 3.10+
- Git
- gh CLI (for GitHub operations)
- Optional: Ollama, MLX, or llama.cpp for local AI

## License

MIT License

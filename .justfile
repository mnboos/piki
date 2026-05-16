set dotenv-load

# Run the full stack (mirrors run.sh)
run:
    ./run.sh

# Generate a CPU flamegraph SVG by sampling the running Django process for 30 s.
# Start 'just run' in another terminal first.
flamegraph:
    mkdir -p logs
    uv run py-spy record \
        --output logs/flamegraph_$(date +%Y%m%dT%H%M%S).svg \
        --duration 30 \
        --pid $(pgrep -f "manage.py runserver")
    @echo "Flamegraph saved to logs/"

# Live top-like profiler view.  Start 'just run' in another terminal first.
profile-live:
    uv run py-spy top --pid $(pgrep -f "manage.py runserver")

# Run the full stack with PIKI_PROFILE=1 to emit per-stage timing lines.
# Output is tee'd to logs/profile_<ts>.log.
profile-staged:
    mkdir -p logs
    PIKI_PROFILE=1 ./run.sh 2>&1 | tee logs/profile_$(date +%Y%m%dT%H%M%S).log

# Parse the most recent profile log and print a p50/p95/p99 summary table.
profile-report:
    uv run python benchmarks/parse_perf_log.py $(ls -t logs/profile_*.log | head -1)

[env("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")]
[env("ANTHROPIC_DEFAULT_HAIKU_MODEL", "deepseek-v4-flash")]
[env("ANTHROPIC_DEFAULT_OPUS_MODEL", "deepseek-v4-pro[1m]")]
[env("ANTHROPIC_DEFAULT_SONNET_MODEL", "deepseek-v4-pro[1m]")]
[env("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")]
[env("CLAUDE_CODE_EFFORT_LEVEL", "max")]
[env("CLAUDE_CODE_SUBAGENT_MODEL", "deepseek-v4-pro[1m]")]
[env("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")]
claude-deepseek:
    claude --model opus --effort max

alias claude := claude-deepseek

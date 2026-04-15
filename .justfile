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

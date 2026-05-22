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

# Compile + install the pump PWM device-tree overlay and enable it in
# /boot/config.txt. Needs sudo; reboot afterwards for it to take effect.
# Pump ENA uses pin 27 (PWM ctrl 34160000); see hardware/overlays/.
install-pump-overlay:
    #!/usr/bin/env bash
    set -euo pipefail
    src=hardware/overlays/dtoverlay_pump_pwm2.dts
    dtc -@ -I dts -O dtb -o /tmp/dtoverlay_pump_pwm2.dtbo "$src"
    sudo cp "$src" /boot/overlays/dtoverlay_pump_pwm2.dts
    sudo cp /tmp/dtoverlay_pump_pwm2.dtbo /boot/overlays/dtoverlay_pump_pwm2.dtbo
    # enable in config.txt (idempotent); must load after dtoverlay_pwm3 (servos)
    if ! grep -q '^dtoverlay=dtoverlay_pump_pwm2$' /boot/config.txt; then
        sudo cp /boot/config.txt /boot/config.txt.bak.$(date +%Y%m%d-%H%M%S)
        echo 'dtoverlay=dtoverlay_pump_pwm2' | sudo tee -a /boot/config.txt >/dev/null
    fi
    echo "Installed. Reboot to activate, then check: ls /sys/class/pwm/"

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

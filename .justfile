set dotenv-load

set windows-powershell := true

_BASE     := "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8"
_MODEL_DIR := "model"
_NAL_DIR  := justfile_directory() / "piki_nal"
_VENV     := justfile_directory() / ".venv"

# Set system timezone to Europe/Zurich and ensure NTP is running
set-timezone:
    sudo timedatectl set-timezone Europe/Zurich
    sudo systemctl enable --now ntp
    timedatectl

# Download the YOLO26 nano segmentation model into the repo's model/ directory
download-seg-model:
    wget -q --show-progress \
        "{{_BASE}}/yolo26n_seg_bayese_640x640_nv12.bin" \
        -O "{{_MODEL_DIR}}/yolo26n_seg_bayese_640x640_nv12.bin"

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

# ── piki_nal Rust extension ──────────────────────────────────────────────────

# Build the piki_nal Rust extension in release mode (~30–90 s first build,
# ~30 s incremental).  Re-run whenever piki_nal/src/ changes.
build-nal:
    #!/usr/bin/env bash
    set -euo pipefail
    source "$HOME/.cargo/env"
    cd "{{_NAL_DIR}}"
    VIRTUAL_ENV="{{_VENV}}" \
        PATH="{{_VENV}}/bin:$PATH" \
        "$HOME/.local/bin/maturin" develop --release

# Fast (unoptimised) piki_nal build for development iteration.
build-nal-dev:
    #!/usr/bin/env bash
    set -euo pipefail
    source "$HOME/.cargo/env"
    cd "{{_NAL_DIR}}"
    VIRTUAL_ENV="{{_VENV}}" \
        PATH="{{_VENV}}/bin:$PATH" \
        "$HOME/.local/bin/maturin" develop

# Type-check piki_nal without linking a .so (fast, no Python needed).
check-nal:
    #!/usr/bin/env bash
    set -euo pipefail
    source "$HOME/.cargo/env"
    cd "{{_NAL_DIR}}"
    cargo check

# Run piki_nal Rust unit tests.
test-nal:
    #!/usr/bin/env bash
    set -euo pipefail
    source "$HOME/.cargo/env"
    cd "{{_NAL_DIR}}"
    cargo test

# ── Production deploy ────────────────────────────────────────────────────────

# Build the Vue frontend to frontend/dist/
build-frontend:
    cd frontend && npm ci && npm run build

# Collect Django static files into src/staticfiles/ (served by Caddy at /static/)
collectstatic:
    cd src && uv run python manage.py collectstatic --noinput

# Full prod build: frontend + staticfiles + Rust extension. Run after pulling changes.
build-prod: build-frontend collectstatic build-nal

# One-time: symlink /etc/caddy/Caddyfile -> deploy/Caddyfile and add the
# caddy user to the sunrise group so it can read the repo files. Backs up
# any pre-existing regular Caddyfile, validates the new config, and reloads
# (or restarts, if the group was just added). Idempotent.
install-caddy-config:
    #!/usr/bin/env bash
    set -euo pipefail
    target=/etc/caddy/Caddyfile
    src=/home/sunrise/src/piki/deploy/Caddyfile
    needs_restart=0

    # Grant caddy read access via group membership.
    if ! id -nG caddy | tr ' ' '\n' | grep -qx sunrise; then
        sudo usermod -aG sunrise caddy
        needs_restart=1
    fi

    # Install symlink, backing up any pre-existing regular file.
    if [ -L "$target" ] && [ "$(readlink "$target")" = "$src" ]; then
        echo "Symlink already in place."
    else
        if [ -e "$target" ] && [ ! -L "$target" ]; then
            sudo cp "$target" "$target.bak.$(date +%Y%m%d-%H%M%S)"
        fi
        sudo ln -sfn "$src" "$target"
    fi

    sudo caddy validate --config "$target" --adapter caddyfile

    # Group changes only apply to processes started AFTER usermod, so restart
    # caddy the first time. Otherwise reload is enough to pick up Caddyfile.
    if [ "$needs_restart" = "1" ]; then
        sudo systemctl restart caddy
    else
        sudo systemctl reload caddy
    fi
    echo "Done. $target -> $src; caddy is in group sunrise."

# One-time: install the systemd unit for the backend.
install-backend-service:
    sudo cp deploy/piki.service /etc/systemd/system/piki.service
    sudo systemctl daemon-reload
    sudo systemctl enable piki.service
    @echo "Installed. Start with:  just serve-start"

install: install-backend-service install-caddy-config

# Reverse of install-caddy-config + install-backend-service:
#   - stop, disable, and remove piki.service
#   - restore /etc/caddy/Caddyfile from the most recent backup
#   - remove caddy from the sunrise group
# Idempotent — safe to re-run.
uninstall:
    #!/usr/bin/env bash
    set -euo pipefail

    # Backend service.
    if systemctl list-unit-files piki.service --no-legend 2>/dev/null | grep -q .; then
        sudo systemctl disable --now piki.service || true
        sudo rm -f /etc/systemd/system/piki.service
        sudo systemctl daemon-reload
        echo "Removed piki.service."
    else
        echo "piki.service not installed; skipping."
    fi

    # Caddyfile symlink → restore latest backup.
    target=/etc/caddy/Caddyfile
    if [ -L "$target" ]; then
        latest_bak=$(ls -1t "$target".bak.* 2>/dev/null | head -n1 || true)
        if [ -n "$latest_bak" ]; then
            sudo rm -f "$target"
            sudo mv "$latest_bak" "$target"
            sudo systemctl reload caddy
            echo "Restored $target from $(basename "$latest_bak")."
        else
            echo "WARNING: $target is a symlink but no backup found; leaving it in place."
            echo "         Remove it manually after providing a replacement Caddyfile."
        fi
    else
        echo "$target is not a symlink; leaving it alone."
    fi

    # Caddy group membership.
    if id -nG caddy 2>/dev/null | tr ' ' '\n' | grep -qx sunrise; then
        sudo gpasswd -d caddy sunrise >/dev/null
        sudo systemctl restart caddy
        echo "Removed caddy from sunrise group."
    fi

    echo "Uninstall complete."

# Validate the Caddyfile syntax without applying it.
caddy-check:
    caddy validate --config deploy/Caddyfile --adapter caddyfile

# Reload Caddy after editing deploy/Caddyfile (no downtime).
caddy-reload:
    sudo systemctl reload caddy

# Start / stop / restart the whole stack.
serve-start:
    sudo systemctl start piki.service
    sudo systemctl reload caddy

serve-stop:
    sudo systemctl stop piki.service

serve-restart:
    sudo systemctl restart piki.service
    sudo systemctl reload caddy

serve-status:
    systemctl status piki.service caddy.service --no-pager

# Tail backend logs (last 50 lines, then follow).
serve-logs:
    journalctl -fu piki.service -n 50 -q

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

run-foxglove:
    #!/usr/bin/env bash
    source /opt/ros/humble/setup.bash
    ros2 launch foxglove_bridge foxglove_bridge_launch.xml

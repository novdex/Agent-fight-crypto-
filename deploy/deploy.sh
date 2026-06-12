#!/usr/bin/env bash
# One-shot VPS deployment for Crypto LLM Arena (Debian/Ubuntu, run as root).
#
#   curl -fsSL <raw-url>/deploy/deploy.sh | bash -s -- <git-repo-url> [branch]
# or, with the repo already cloned:
#   sudo ./deploy/deploy.sh <git-repo-url> [branch]
#
# Installs to /opt/crypto-llm-arena under a dedicated 'arena' system user and
# runs `arena loop --interval-mins 60` as a systemd service.
set -euo pipefail

REPO_URL="${1:?usage: deploy.sh <git-repo-url> [branch]}"
BRANCH="${2:-main}"
APP_DIR=/opt/crypto-llm-arena

command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }
command -v python3 >/dev/null || { apt-get update -qq && apt-get install -y -qq python3 python3-venv; }
python3 -c "import venv" 2>/dev/null || apt-get install -y -qq python3-venv

id -u arena >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin arena

if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" fetch origin "$BRANCH"
    git -C "$APP_DIR" checkout "$BRANCH"
    git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q -U pip
"$APP_DIR/.venv/bin/pip" install -q -e "$APP_DIR"

if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    chmod 600 "$APP_DIR/.env"
    NEEDS_KEYS=1
fi

chown -R arena:arena "$APP_DIR"
install -m 644 "$APP_DIR/deploy/arena.service" /etc/systemd/system/arena.service
systemctl daemon-reload
systemctl enable arena >/dev/null

if [ "${NEEDS_KEYS:-0}" = "1" ]; then
    echo
    echo ">> Now add your API keys to $APP_DIR/.env, then run: systemctl restart arena"
else
    systemctl restart arena
    sleep 2
    systemctl status arena --no-pager -l | head -12
fi

echo
echo "Useful commands:"
echo "  journalctl -u arena -f                                   # live logs"
echo "  sudo -u arena $APP_DIR/.venv/bin/arena leaderboard        # standings"
echo "  systemctl stop|start|restart arena"

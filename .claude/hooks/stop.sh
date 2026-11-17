#!/usr/bin/env bash
# Stop hook: format with ruff, then run pytest. Failures are sent back to Claude
# (exit 2) so it fixes them, except when already continuing from this hook.
cd "$CLAUDE_PROJECT_DIR" || exit 0
input=$(cat)
active=$(jq -r '.stop_hook_active // false' <<<"$input")

uv run --quiet ruff format . >/dev/null 2>&1

if ! out=$(uv run --quiet pytest -q 2>&1); then
  [ "$active" = "true" ] && { echo "pytest still failing" >&2; exit 0; }
  echo "pytest failed — fix before stopping:" >&2
  tail -n 40 <<<"$out" >&2
  exit 2
fi
exit 0

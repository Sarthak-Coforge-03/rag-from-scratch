#!/usr/bin/env bash
# PreToolUse hook: deny any tool call that touches a .env file
# (.env, .env.local, …; .env.example is allowed).
input=$(cat)
target=$(jq -r '.tool_input | (.file_path // .path // .notebook_path // .command // .pattern // "")' <<<"$input")
re='(^|[^A-Za-z0-9_.-])\.env(\.[A-Za-z0-9_-]+)?($|[^A-Za-z0-9_.-])'
if grep -Eq "$re" <<<"$target" && ! grep -Eq '\.env\.example' <<<"$target"; then
  jq -n '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",
    permissionDecisionReason:"Access to .env files is blocked by project hook (secrets)."}}'
fi
exit 0

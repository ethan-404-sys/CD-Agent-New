#!/usr/bin/env bash
# Manage isolated Git worktrees for parallel CD-Agent tracks (T3 / multimodal / verifier).
# Automatically links shared models/, datasets/, third_party/, and outputs/ from the main repo.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WT_ROOT="${CD_AGENT_WT_ROOT:-$(dirname "$REPO")/CD-Agent-wt}"

usage() {
    cat <<'EOF'
Usage:
  scripts/worktree.sh create <name> [branch]   # Create ../CD-Agent-wt/<name> on feat/<name> (or [branch])
  scripts/worktree.sh remove <name>            # Cleanly remove worktree (does not force-delete dirty changes)
  scripts/worktree.sh list                     # List active git worktrees
EOF
}

cmd="${1:-}"
case "$cmd" in
    create)
        name="${2:?Missing worktree name (e.g. t3 or multimodal)}"
        branch="${3:-feat/$name}"
        target="$WT_ROOT/$name"
        mkdir -p "$WT_ROOT"
        if git -C "$REPO" show-ref --verify --quiet "refs/heads/$branch"; then
            git -C "$REPO" worktree add "$target" "$branch"
        else
            git -C "$REPO" worktree add -b "$branch" "$target" main
        fi
        for dir in models datasets third_party outputs; do
            rm -rf "$target/$dir"
            ln -s "$REPO/$dir" "$target/$dir"
        done
        mkdir -p "$target/.runtime"
        echo "Worktree ready: $target (branch=$branch)"
        echo "Linked shared dirs: models, datasets, third_party, outputs -> $REPO"
        ;;
    remove)
        name="${2:?Missing worktree name}"
        target="$WT_ROOT/$name"
        for dir in models datasets third_party outputs; do
            [[ -L "$target/$dir" ]] && rm -f "$target/$dir"
        done
        git -C "$REPO" worktree remove "$target"
        git -C "$REPO" worktree prune
        echo "Removed worktree: $target"
        ;;
    list)
        git -C "$REPO" worktree list
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

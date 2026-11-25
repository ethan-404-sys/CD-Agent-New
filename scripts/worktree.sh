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
        # Keep tracked README.md files intact in models/, datasets/, outputs/ so git status stays clean.
        for dir in models datasets outputs; do
            mkdir -p "$target/$dir"
            for item in "$REPO/$dir"/*; do
                [[ -e "$item" || -L "$item" ]] || continue
                base="$(basename "$item")"
                [[ "$base" == "README.md" ]] && continue
                rm -rf "$target/$dir/$base"
                ln -s "$item" "$target/$dir/$base"
            done
        done
        rm -rf "$target/third_party"
        ln -s "$REPO/third_party" "$target/third_party"
        mkdir -p "$target/.runtime"
        echo "Worktree ready: $target (branch=$branch)"
        echo "Linked shared entries: models/*, datasets/*, outputs/*, third_party -> $REPO"
        ;;
    remove)
        name="${2:?Missing worktree name}"
        target="$WT_ROOT/$name"
        # Refuse to silently drop real non-symlink outputs created only inside this worktree
        if [[ -d "$target/outputs" && ! -L "$target/outputs" ]]; then
            for item in "$target/outputs"/*; do
                [[ -e "$item" || -L "$item" ]] || continue
                base="$(basename "$item")"
                [[ "$base" == "README.md" ]] && continue
                if [[ ! -L "$item" ]]; then
                    echo "Refusing to remove worktree: real output exists at $item; move it into $REPO/outputs/ first." >&2
                    exit 1
                fi
            done
        fi
        for dir in models datasets outputs; do
            if [[ -L "$target/$dir" ]]; then
                rm -f "$target/$dir"
            elif [[ -d "$target/$dir" ]]; then
                for item in "$target/$dir"/*; do
                    [[ -L "$item" ]] && rm -f "$item"
                done
            fi
        done
        [[ -L "$target/third_party" ]] && rm -f "$target/third_party"
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

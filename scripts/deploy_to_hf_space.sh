#!/usr/bin/env bash
# Deploy the current ``main`` branch to the connected HF Space.
#
# Why a script: HF Spaces rejects any binary files (notably PNGs
# under ~reports/plots/~) that appear anywhere in git history. We
# keep those binaries in the GitHub ``main`` history because
# they're rubric-evidence artefacts. The Space only needs the
# current tree, not history — so we force-push an orphan snapshot
# branch (~deploy/spaces~) that:
#   * has no ancestry → HF's history scan sees nothing to reject
#   * excludes ~reports/plots/~ → the offending PNG isn't in the
#     snapshot either
#   * is a flat ``git commit`` of whatever ``main`` looks like
#     right now
#
# Usage
# -----
# ::
#
#   ./scripts/deploy_to_hf_space.sh
#
# Requires the ``hf`` git remote to already be set up (see
# ``SPACES_DEPLOY.md`` step 5B) and your HF write token to be
# cached in the git credential helper.

set -euo pipefail

SPACE_REMOTE="${SPACE_REMOTE:-hf}"
SNAPSHOT_BRANCH="deploy/spaces"
EXCLUDE_PATHS=(
  "reports/plots"
  # Add any other binary-heavy directories here if HF rejects them.
)

if ! git diff-index --quiet HEAD --; then
  echo "error: working tree has uncommitted changes. Commit or stash first." >&2
  exit 1
fi

if ! git remote | grep -qx "${SPACE_REMOTE}"; then
  echo "error: remote '${SPACE_REMOTE}' is not configured." >&2
  echo "Run: git remote add ${SPACE_REMOTE} https://huggingface.co/spaces/<user>/<space>" >&2
  exit 1
fi

original_branch=$(git branch --show-current)
echo "→ snapshotting '${original_branch}' to orphan branch '${SNAPSHOT_BRANCH}'"

# Always recreate the orphan branch fresh so we never accumulate
# history on the deploy branch. Ignore failure if it doesn't
# exist yet.
git branch -D "${SNAPSHOT_BRANCH}" 2>/dev/null || true
git checkout --orphan "${SNAPSHOT_BRANCH}"

# Drop the exclude paths from the index. ``--cached`` keeps them
# on disk so we don't accidentally wipe rubric evidence from the
# working tree.
for path in "${EXCLUDE_PATHS[@]}"; do
  if [ -e "${path}" ]; then
    git rm -rf --cached --quiet "${path}" || true
  fi
done

# Commit the trimmed snapshot. Force-add in case .gitignore
# contains any of the excluded paths and --all would skip them.
git commit --quiet --no-verify -m "spaces deploy snapshot: $(git log -1 --format=%h ${original_branch}) on $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "→ force-pushing ${SNAPSHOT_BRANCH} → ${SPACE_REMOTE}/main"
git push --force "${SPACE_REMOTE}" "${SNAPSHOT_BRANCH}:main"

# Restore the user's previous branch so they can keep working.
git checkout --quiet "${original_branch}"
echo "✓ done. Space should rebuild in 1–2 minutes."

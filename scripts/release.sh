#!/usr/bin/env bash
# Release helper for hearable.
#
#   ./scripts/release.sh v0.1.0
#
# Publishes a version end-to-end, in the one order that keeps the Homebrew
# formula's pinned sha256 consistent with what GitHub actually serves:
#
#   1. push master
#   2. (re)point the tag at the CURRENT HEAD and push it          <- so the release
#                                                                    includes every fix
#   3. download GitHub's generated source tarball, compute sha256
#   4. patch packaging/homebrew/hearable.rb (stable url + sha256)
#   5. commit + push that formula bump (lands AFTER the tag, by design)
#   6. mirror the formula into a sibling ../homebrew-hearable tap (if present)
#   7. create the GitHub Release with notes from CHANGELOG.md
#
# Requires: git, gh (authenticated: `gh auth status`), curl, shasum|sha256sum, perl.
set -euo pipefail

VERSION="${1:?usage: scripts/release.sh vX.Y.Z}"
[[ "$VERSION" == v*.*.* ]] || { echo "version must look like v0.1.0"; exit 1; }

REPO="alonecandies/hearable"
FORMULA="packaging/homebrew/hearable.rb"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v gh   >/dev/null || { echo "gh CLI not found"; exit 1; }
gh auth status  >/dev/null 2>&1 || { echo "gh not authenticated — run: gh auth login"; exit 1; }
git diff --quiet && git diff --cached --quiet || {
  echo "working tree is dirty — commit or stash first (the script commits a formula bump)"; exit 1; }

sha256_of() {  # portable: macOS shasum, else GNU sha256sum
  if command -v shasum >/dev/null; then shasum -a 256 "$1" | awk '{print $1}';
  else sha256sum "$1" | awk '{print $1}'; fi
}

echo "==> 1/7 pushing master"
git push origin master

echo "==> 2/7 pointing $VERSION at $(git rev-parse --short HEAD) and pushing the tag"
git tag -f -a "$VERSION" -m "hearable $VERSION"
git push -f origin "$VERSION"

echo "==> 3/7 computing sha256 of the generated source tarball"
URL="https://github.com/$REPO/archive/refs/tags/$VERSION.tar.gz"
TARBALL="$(mktemp -t hearable-src).tar.gz"
SHA=""
for i in $(seq 1 12); do
  if curl -fsSL "$URL" -o "$TARBALL"; then SHA="$(sha256_of "$TARBALL")"; break; fi
  echo "   tarball not ready (GitHub still generating it); retry $i/12"; sleep 3
done
[[ -n "$SHA" ]] || { echo "could not download $URL"; exit 1; }
echo "    sha256 = $SHA"

echo "==> 4/7 patching $FORMULA (stable url + sha256)"
perl -0777 -pi -e \
  "s{(stable do\s*\n\s*url \")[^\"]+}{\${1}https://github.com/$REPO/archive/refs/tags/$VERSION.tar.gz}" \
  "$FORMULA"
perl -0777 -pi -e \
  "s{(stable do[\s\S]*?sha256 \")[0-9a-f]{64}}{\${1}$SHA}" \
  "$FORMULA"
grep -q "$SHA" "$FORMULA" || { echo "formula patch failed — check $FORMULA by hand"; exit 1; }

echo "==> 5/7 committing + pushing the formula bump"
git commit -am "release: $VERSION Homebrew formula sha256"
git push origin master

TAP="$ROOT/../homebrew-hearable"
if [[ -d "$TAP/.git" ]]; then
  echo "==> 6/7 mirroring formula into $TAP"
  mkdir -p "$TAP/Formula"
  cp "$FORMULA" "$TAP/Formula/hearable.rb"
  git -C "$TAP" add Formula/hearable.rb
  git -C "$TAP" commit -m "hearable $VERSION" && git -C "$TAP" push
else
  echo "==> 6/7 no tap checkout at $TAP — set it up once (see end of output)"
fi

echo "==> 7/7 creating GitHub Release $VERSION"
gh release create "$VERSION" --repo "$REPO" \
  --title "hearable $VERSION" --notes-file CHANGELOG.md --verify-tag

echo
echo "Released $VERSION  ->  https://github.com/$REPO/releases/tag/$VERSION"
if [[ ! -d "$TAP/.git" ]]; then
  cat <<EOF

One-time Homebrew tap setup (run from $ROOT/..):
  gh repo create $REPO-tap --public   # creates alonecandies/homebrew-hearable? -> use exact name below
  # NOTE: the repo MUST be named 'homebrew-hearable' for 'brew tap alonecandies/hearable' to work:
  gh repo create alonecandies/homebrew-hearable --public --clone
  mkdir -p homebrew-hearable/Formula
  cp "$FORMULA" homebrew-hearable/Formula/hearable.rb
  cd homebrew-hearable && git add . && git commit -m "hearable $VERSION" && git push && cd -

Then anyone can:  brew tap alonecandies/hearable && brew install hearable
EOF
fi

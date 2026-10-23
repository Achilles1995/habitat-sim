#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Publish CI logs/outputs to a throwaway branch of this repo, so they can be
# read with plain git when the Actions log/artifact storage isn't reachable.
#
# usage: publish_ci_logs.sh BRANCH PATH [PATH...]
#   Each PATH (file or directory) that exists is copied into the branch root.
#   Files named *build-log.txt are also summarized (errors.txt, tail.txt) and
#   gzipped.
set -u
branch="$1"
shift
out="$RUNNER_TEMP/publish-$branch"
rm -rf "$out" && mkdir -p "$out"
for p in "$@"; do
  [ -e "$p" ] || continue
  case "$p" in
    *build-log.txt)
      grep -nE "error|FAILED|CMake Error|fatal" "$p" \
        | grep -viE "warning|-Werror|error_code|errors\.h|_error\.|Error\.cpp|Error\.h|/WX" \
        | head -400 > "$out/errors.txt" || true
      tail -n 3000 "$p" > "$out/tail.txt"
      gzip -c "$p" > "$out/$(basename "$p").gz"
      ;;
    *)
      cp -r "$p" "$out/"
      ;;
  esac
done
cd "$out"
echo "run $GITHUB_RUN_ID sha $GITHUB_SHA job $GITHUB_JOB" > RUN.txt
git init -q && git checkout -q -b tmp
git config user.name "ci" && git config user.email "ci@users.noreply.github.com"
git add -A && git commit -qm "$branch for run $GITHUB_RUN_ID ($GITHUB_SHA)"
git push -f -q "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}" "tmp:$branch"

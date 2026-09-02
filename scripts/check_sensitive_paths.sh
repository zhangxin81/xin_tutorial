#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/check_sensitive_paths.sh        # scan staged files
  scripts/check_sensitive_paths.sh --all  # scan tracked and untracked files

The check rejects common environment-specific or sensitive strings before they
enter commits: personal workspace paths, home directories, worker ids, pod IPs,
and generated trial names.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

declare -a files=()
if [[ "${1:-}" == "--all" ]]; then
  while IFS= read -r -d '' path; do
    files+=("${path}")
  done < <(git ls-files --cached --others --exclude-standard -z)
elif [[ $# -eq 0 ]]; then
  while IFS= read -r -d '' path; do
    files+=("${path}")
  done < <(git diff --cached --name-only --diff-filter=ACMR -z)
else
  usage >&2
  exit 2
fi

if [[ ${#files[@]} -eq 0 ]]; then
  exit 0
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

patterns_file="${tmp_dir}/patterns.txt"
{
  # Any absolute path that includes a per-user directory component.
  printf '%s\n' '(^|[[:space:]`"'\''=:,(])/[[:alnum:]_.-]+([^[:space:]`"'\'')]*[[:alnum:]_.-]+)?/users/[^[:space:]`"'\'')]+'

  # Common home-directory paths without spelling them as example values.
  printf '%s\n' '(^|[[:space:]`"'\''=:,(])/[Hh][Oo][Mm][Ee]/[^/[:space:]`"'\'')]+(/|$)'

  # Worker/runtime identifiers that should be replaced by placeholders.
  printf '%s\n' '[0-9a-fA-F]{4}(:[0-9a-fA-F]{0,4}){2,}'
  printf '%s\n' 'trial-[0-9]+-trialrun-[0-9]+'
  printf '%s\n' 'podIP[[:space:]]+[0-9a-fA-F:]+'
  printf '%s\n' 'worker[ _-]?(id)?[^[:alnum:]]*[0-9]{6,}'
} >"${patterns_file}"

is_text() {
  local mode="$1"
  local path="$2"
  if [[ "${mode}" == "--all" ]]; then
    LC_ALL=C grep -Iq . "${path}"
  else
    git show ":${path}" | LC_ALL=C grep -Iq .
  fi
}

scan_file() {
  local mode="$1"
  local path="$2"
  if [[ "${mode}" == "--all" ]]; then
    grep -nE -f "${patterns_file}" -- "${path}" || true
  else
    git show ":${path}" | grep -nE -f "${patterns_file}" || true
  fi
}

failed=0
for path in "${files[@]}"; do
  [[ -f "${path}" || "${1:-}" != "--all" ]] || continue

  if ! is_text "${1:---staged}" "${path}"; then
    continue
  fi

  matches="$(scan_file "${1:---staged}" "${path}")"
  if [[ -n "${matches}" ]]; then
    if [[ ${failed} -eq 0 ]]; then
      cat >&2 <<'EOF'
Sensitive information check failed.

Remove or replace environment-specific values before committing. Prefer
relative paths or placeholders such as <tutorial-root>, <run_name>,
<worker-id>, and <pod-ip>.

Matches:
EOF
    fi
    while IFS= read -r line; do
      printf '  %s:%s\n' "${path}" "${line}" >&2
    done <<<"${matches}"
    failed=1
  fi
done

exit "${failed}"

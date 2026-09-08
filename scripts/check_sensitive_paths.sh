#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/check_sensitive_paths.sh [--virtual-prefix <prefix>]...  # scan staged files
  scripts/check_sensitive_paths.sh --all [--virtual-prefix <prefix>]...

The check rejects common environment-specific or sensitive strings before they
enter commits: personal workspace paths, home directories, worker ids, pod IPs,
and generated trial names.

Every other environment-specific absolute path must be rewritten under a
virtual prefix. The default prefix is /path/to; replace it with
--virtual-prefix <prefix> (repeatable) or the space-separated XIN_VIRTUAL_PREFIX
environment variable. Generic system paths such as /tmp, /usr, /etc, and /opt
are always allowed.
EOF
}

declare -a virtual_prefix_args=()
scan_all=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --all)
      scan_all=1
      shift
      ;;
    --virtual-prefix)
      if [[ $# -lt 2 ]]; then
        usage >&2
        exit 2
      fi
      virtual_prefix_args+=("$2")
      shift 2
      ;;
    --virtual-prefix=*)
      virtual_prefix_args+=("${1#--virtual-prefix=}")
      shift
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

declare -a virtual_prefixes=()

add_virtual_prefix() {
  local prefix="$1"
  while [[ "${prefix}" == "/"* ]]; do
    prefix="${prefix#/}"
  done
  while [[ "${prefix}" == *"/" ]]; do
    prefix="${prefix%/}"
  done
  if [[ -n "${prefix}" ]]; then
    virtual_prefixes+=("/${prefix}")
  fi
}

if [[ ${#virtual_prefix_args[@]} -eq 0 && -n "${XIN_VIRTUAL_PREFIX:-}" ]]; then
  read -r -a virtual_prefix_args <<<"${XIN_VIRTUAL_PREFIX}"
fi
if [[ ${#virtual_prefix_args[@]} -eq 0 ]]; then
  virtual_prefix_args=("/path/to")
fi
for prefix in "${virtual_prefix_args[@]}"; do
  add_virtual_prefix "${prefix}"
done

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

declare -a files=()
if [[ "${scan_all}" -eq 1 ]]; then
  while IFS= read -r -d '' path; do
    files+=("${path}")
  done < <(git ls-files --cached --others --exclude-standard -z)
else
  while IFS= read -r -d '' path; do
    files+=("${path}")
  done < <(git diff --cached --name-only --diff-filter=ACMR -z)
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

# Any multi-segment absolute path; filter_unrewritten_paths() then keeps only
# the ones that are neither generic system paths nor virtual-prefix paths.
absolute_path_pattern='(^|[[:space:]`"'\''=:,(])/[[:alnum:]_.-]+(/[[:alnum:]_.-]+)+'

is_allowed_path() {
  local candidate="$1"
  local prefix
  case "${candidate}" in
    /tmp/*|/usr/*|/etc/*|/var/*|/opt/*|/bin/*|/sbin/*|/lib/*|/lib32/*|/lib64/*|/dev/*|/proc/*|/sys/*|/run/*)
      return 0
      ;;
  esac
  if [[ ${#virtual_prefixes[@]} -gt 0 ]]; then
    for prefix in "${virtual_prefixes[@]}"; do
      case "${candidate}" in
        "${prefix}"|"${prefix}"/*)
          return 0
          ;;
      esac
    done
  fi
  return 1
}

# Input lines are "<line-no>:<match>" from grep -no, where <match> starts with
# the separator character that preceded the path.
filter_unrewritten_paths() {
  local line
  local number
  local candidate
  while IFS= read -r line; do
    number="${line%%:*}"
    line="${line#*:}"
    candidate="/${line#*/}"
    if ! is_allowed_path "${candidate}"; then
      printf '%s:%s\n' "${number}" "${candidate}"
    fi
  done
}

is_text() {
  local mode="$1"
  local path="$2"
  if [[ "${mode}" == "all" ]]; then
    LC_ALL=C grep -Iq . "${path}"
  else
    git show ":${path}" | LC_ALL=C grep -Iq .
  fi
}

scan_file() {
  local mode="$1"
  local path="$2"
  if [[ "${mode}" == "all" ]]; then
    {
      grep -nE -f "${patterns_file}" -- "${path}" || true
      grep -noE "${absolute_path_pattern}" -- "${path}" | filter_unrewritten_paths || true
    }
  else
    {
      git show ":${path}" | grep -nE -f "${patterns_file}" || true
      git show ":${path}" | grep -noE "${absolute_path_pattern}" | filter_unrewritten_paths || true
    }
  fi
}

mode="staged"
if [[ "${scan_all}" -eq 1 ]]; then
  mode="all"
fi

if [[ ${#virtual_prefixes[@]} -gt 0 ]]; then
  prefix_list="${virtual_prefixes[*]}"
else
  prefix_list="none configured"
fi

failed=0
for path in "${files[@]}"; do
  if [[ "${mode}" == "all" && ! -f "${path}" ]]; then
    continue
  fi

  if ! is_text "${mode}" "${path}"; then
    continue
  fi

  matches="$(scan_file "${mode}" "${path}")"
  if [[ -n "${matches}" ]]; then
    if [[ ${failed} -eq 0 ]]; then
      {
        printf 'Sensitive information check failed.\n\n'
        printf 'Remove or replace environment-specific values before committing. Rewrite\n'
        printf 'absolute paths under the virtual prefix (%s), or use relative paths or\n' "${prefix_list}"
        printf 'placeholders such as <run_name>, <worker-id>, and <pod-ip>.\n\n'
        printf 'Matches:\n'
      } >&2
    fi
    while IFS= read -r line; do
      printf '  %s:%s\n' "${path}" "${line}" >&2
    done <<<"${matches}"
    failed=1
  fi
done

exit "${failed}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_IMAGE="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
DEFAULT_MODEL="Qwen/Qwen3.5-2B"
DEFAULT_BENCHMARKS="coding,reasoning"
DEFAULT_WEIGHTS="training_results/code_embedded/qwen35-2b/coprocessor_weights.pt"
DEFAULT_OUTPUT="training_results/code_embedded/qwen35-2b/realworld_benchmark_vast.json"
DEFAULT_REMOTE_DIR="/workspace/ncpu_vast_benchmark"
DEFAULT_DISK_GB="30"
DEFAULT_BOOT_TIMEOUT="900"
DEFAULT_POLL_SECONDS="15"
DEFAULT_MAX_OFFERS_PER_QUERY="3"

usage() {
  cat <<'EOF'
Usage:
  benchmarks/run_vast_coprocessor_benchmark.sh [options]

Launch the coprocessor real-world benchmark on a Vast.ai SSH instance, upload
the minimal benchmark bundle, and start the run under nohup.

Important:
  This launcher always creates the instance with `--ssh`. Creating a Vast
  instance without `--ssh` still gives you an ssh gateway URL, but Vast does
  not inject ssh into the container, so public-key auth never succeeds.

Options:
  --offer-id ID              Use a specific Vast offer ID instead of searching.
  --query QUERY              Search query to use for offers. Can be repeated.
  --image IMAGE              Docker image. Default:
                             pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
  --disk GB                  Instance disk size in GB. Default: 30
  --model MODEL              Hugging Face model ID. Default: Qwen/Qwen3.5-2B
  --weights PATH             Local weights path relative to repo or absolute.
  --humaneval-path PATH      Local HumanEval.jsonl path if humaneval is enabled.
  --benchmarks LIST          Benchmark list passed through to the Python runner.
                             Default: coding,reasoning
  --output PATH              Output JSON path inside the repo tree.
  --remote-dir PATH          Remote working directory. Default:
                             /workspace/ncpu_vast_benchmark
  --boot-timeout SEC         Max seconds to wait for SSH readiness. Default: 900
  --wait                     Wait for benchmark completion, then download output.
  --destroy-on-success       Destroy the instance after a successful waited run.
  --keep-failed-instance     Do not destroy a failed boot candidate automatically.
  --help                     Show this help text.

Environment overrides:
  VAST_BIN                   Override the vastai executable path.
  SSH_PRIVATE_KEY            Private key for ssh/scp. Default: ~/.ssh/id_rsa
  SSH_PUBLIC_KEY             Public key to attach. Default: ${SSH_PRIVATE_KEY}.pub
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

resolve_vast_bin() {
  if [[ -n "${VAST_BIN:-}" ]]; then
    printf '%s\n' "${VAST_BIN}"
    return
  fi
  if command -v vastai >/dev/null 2>&1; then
    command -v vastai
    return
  fi
  local candidate
  for candidate in \
    "${HOME}/Library/Python/3.14/bin/vastai" \
    "${HOME}/Library/Python/3.13/bin/vastai" \
    "${HOME}/Library/Python/3.12/bin/vastai"
  do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  die "Could not find vastai. Set VAST_BIN=/absolute/path/to/vastai."
}

read_instance_fields() {
  local instance_id="$1"
  local raw_output
  raw_output="$("${VAST_BIN}" show instance "${instance_id}" --raw 2>/dev/null)"
  INSTANCE_RAW="${raw_output}" python3 - <<'PY'
import ast
import json
import os
import sys

text = os.environ.get("INSTANCE_RAW", "").strip()
obj = None
for parser in (json.loads, ast.literal_eval):
    try:
        obj = parser(text)
        break
    except Exception:
        continue

if not isinstance(obj, dict):
    sys.exit(1)

fields = [
    obj.get("actual_status") or "",
    obj.get("cur_state") or "",
    obj.get("ssh_host") or "",
    str(obj.get("ssh_port") or ""),
    (obj.get("status_msg") or "").replace("\r", " ").replace("\n", " "),
]
print("\t".join(fields))
PY
}

contains_benchmark() {
  local name="$1"
  IFS=',' read -r -a current_benchmarks <<<"${BENCHMARKS}"
  local item
  for item in "${current_benchmarks[@]}"; do
    if [[ "${item}" == "${name}" ]]; then
      return 0
    fi
  done
  return 1
}

ensure_file_exists() {
  local path="$1"
  [[ -e "${path}" ]] || die "Required file not found: ${path}"
}

make_relative_to_repo() {
  local abs_path="$1"
  case "${abs_path}" in
    "${PROJECT_ROOT}"/*)
      printf '%s\n' "${abs_path#${PROJECT_ROOT}/}"
      ;;
    *)
      die "Path must be inside the repo: ${abs_path}"
      ;;
  esac
}

build_bundle() {
  local bundle_path="$1"
  local weight_rel human_eval_rel
  local -a bundle_items

  weight_rel="$(make_relative_to_repo "${WEIGHTS_PATH}")"

  bundle_items=(
    "benchmarks/benchmark_coprocessor_realworld.py"
    "ncpu/coprocessor"
    "models/alu"
    "${weight_rel}"
  )

  if contains_benchmark "humaneval"; then
    human_eval_rel="${HUMANEVAL_PATH_REL}"
    [[ -n "${human_eval_rel}" ]] || die "--benchmarks includes humaneval but no HumanEval.jsonl path was supplied or found."
    bundle_items+=("${human_eval_rel}")
  fi

  mkdir -p "$(dirname "${bundle_path}")"
  log "Building benchmark bundle at ${bundle_path}"
  (
    cd "${PROJECT_ROOT}"
    tar -czf "${bundle_path}" "${bundle_items[@]}"
  )
}

offer_ids_from_query() {
  local query="$1"
  "${VAST_BIN}" search offers "${query}" 2>/dev/null \
    | awk 'NR > 1 && $1 ~ /^[0-9]+$/ {print $1}' \
    | head -n "${MAX_OFFERS_PER_QUERY}"
}

destroy_instance() {
  local instance_id="$1"
  if [[ -n "${instance_id}" ]]; then
    log "Destroying instance ${instance_id}"
    "${VAST_BIN}" destroy instance "${instance_id}" >/dev/null 2>&1 || true
  fi
}

attach_ssh_key() {
  local instance_id="$1"
  local pubkey
  pubkey="$(<"${SSH_PUBLIC_KEY}")"
  "${VAST_BIN}" attach ssh "${instance_id}" "${pubkey}" >/dev/null 2>&1 || true
}

ssh_base_cmd() {
  local host="$1"
  local port="$2"
  shift 2 || true
  ssh \
    -F /dev/null \
    -i "${SSH_PRIVATE_KEY}" \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p "${port}" \
    "root@${host}" \
    "$@"
}

scp_to_remote() {
  local local_path="$1"
  local host="$2"
  local port="$3"
  local remote_path="$4"
  scp \
    -F /dev/null \
    -i "${SSH_PRIVATE_KEY}" \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -P "${port}" \
    "${local_path}" \
    "root@${host}:${remote_path}"
}

scp_from_remote() {
  local host="$1"
  local port="$2"
  local remote_path="$3"
  local local_path="$4"
  mkdir -p "$(dirname "${local_path}")"
  scp \
    -F /dev/null \
    -i "${SSH_PRIVATE_KEY}" \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -P "${port}" \
    "root@${host}:${remote_path}" \
    "${local_path}"
}

create_instance_from_offer() {
  local offer_id="$1"
  local create_output instance_id

  log "Creating Vast instance from offer ${offer_id}"
  create_output="$(
    "${VAST_BIN}" create instance "${offer_id}" \
      --image "${IMAGE}" \
      --disk "${DISK_GB}" \
      --ssh \
      --cancel-unavail \
      --lang-utf8 \
      2>&1
  )" || return 1

  instance_id="$(
    CREATE_OUTPUT="${create_output}" python3 - <<'PY'
import ast
import json
import os
import re
import sys

text = os.environ.get("CREATE_OUTPUT", "").strip()
for parser in (json.loads, ast.literal_eval):
    try:
        obj = parser(text)
        if isinstance(obj, dict) and obj.get("new_contract"):
            print(obj["new_contract"])
            raise SystemExit(0)
    except Exception:
        pass

match = re.search(r"new_contract['\"]?\s*:\s*([0-9]+)", text)
if match:
    print(match.group(1))
    raise SystemExit(0)
sys.exit(1)
PY
  )" || {
    printf '%s\n' "${create_output}" >&2
    return 1
  }

  printf '%s\n' "${instance_id}"
}

wait_for_ssh_ready() {
  local instance_id="$1"
  local deadline actual_status cur_state host port status_msg
  deadline=$((SECONDS + BOOT_TIMEOUT))

  attach_ssh_key "${instance_id}"

  while (( SECONDS < deadline )); do
    if IFS=$'\t' read -r actual_status cur_state host port status_msg < <(read_instance_fields "${instance_id}"); then
      if [[ "${status_msg}" == *"failed to inject CDI"* || "${status_msg}" == *"OCI runtime create failed"* ]]; then
        log "Instance ${instance_id} hit a GPU/container startup failure: ${status_msg}"
        return 2
      fi

      log "Instance ${instance_id}: actual=${actual_status:-?} cur=${cur_state:-?} ssh=${host:-?}:${port:-?}"

      if [[ -n "${host}" && -n "${port}" ]]; then
        if ssh_base_cmd "${host}" "${port}" "printf READY" >/dev/null 2>&1; then
          READY_HOST="${host}"
          READY_PORT="${port}"
          return 0
        fi
      fi
    fi
    sleep "${POLL_SECONDS}"
  done

  return 1
}

start_remote_benchmark() {
  local host="$1"
  local port="$2"
  local bundle_name remote_bundle remote_log remote_done remote_script_local remote_script_remote

  bundle_name="$(basename "${BUNDLE_PATH}")"
  remote_bundle="${REMOTE_DIR}/${bundle_name}"
  remote_log="${REMOTE_DIR}/benchmark.log"
  remote_done="${REMOTE_DIR}/benchmark.done"
  remote_script_remote="${REMOTE_DIR}/run_benchmark.sh"

  log "Creating remote workdir ${REMOTE_DIR}"
  ssh_base_cmd "${host}" "${port}" "mkdir -p '${REMOTE_DIR}'"

  log "Uploading bundle"
  scp_to_remote "${BUNDLE_PATH}" "${host}" "${port}" "${remote_bundle}"

  remote_script_local="$(mktemp "${TMPDIR:-/tmp}/ncpu_vast_remote.XXXXXX.sh")"
  cat >"${remote_script_local}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "${REMOTE_DIR}"
tar -xzf "${bundle_name}"
mkdir -p "\$(dirname "${OUTPUT_REL}")"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 "torch>=2.4.0" "torchvision>=0.20.1"
python3 -m pip install --upgrade "transformers>=5.3.0" "datasets>=2.14.0" "accelerate>=0.24.0" "peft>=0.6.0" "tqdm>=4.66.0"

status=0
benchmark_args=(
  --model "${MODEL}"
  --coprocessor-weights "${WEIGHTS_REL}"
  --models-dir models
  --benchmarks "${BENCHMARKS}"
  --output "${OUTPUT_REL}"
)

if [[ -n "${HUMANEVAL_PATH_REL}" ]]; then
  benchmark_args+=(--humaneval-path "${HUMANEVAL_PATH_REL}")
fi

python3 benchmarks/benchmark_coprocessor_realworld.py "\${benchmark_args[@]}" || status=\$?

printf '%s\n' "\${status}" > "${REMOTE_DIR}/benchmark.exit"
touch "${remote_done}"
exit "\${status}"
EOF

  log "Uploading remote launcher"
  scp_to_remote "${remote_script_local}" "${host}" "${port}" "${remote_script_remote}"
  rm -f "${remote_script_local}"

  log "Launching remote benchmark"
  ssh_base_cmd "${host}" "${port}" "chmod +x '${remote_script_remote}' && nohup '${remote_script_remote}' >'${remote_log}' 2>&1 </dev/null & echo \$! > '${REMOTE_DIR}/benchmark.pid'"

  printf '%s\n' "${remote_log}"
}

wait_for_remote_completion() {
  local host="$1"
  local port="$2"
  local remote_log="$3"
  local remote_done="${REMOTE_DIR}/benchmark.done"
  local remote_exit="${REMOTE_DIR}/benchmark.exit"
  local remote_output="${REMOTE_DIR}/${OUTPUT_REL}"
  local exit_code

  log "Waiting for remote benchmark completion"
  while true; do
    if ssh_base_cmd "${host}" "${port}" "test -f '${remote_done}'"; then
      break
    fi
    log "Remote benchmark still running"
    ssh_base_cmd "${host}" "${port}" "tail -n 20 '${remote_log}' 2>/dev/null || true" || true
    sleep 30
  done

  exit_code="$(ssh_base_cmd "${host}" "${port}" "cat '${remote_exit}' 2>/dev/null || printf '1'")"

  log "Downloading benchmark output"
  if [[ "${exit_code}" == "0" ]]; then
    scp_from_remote "${host}" "${port}" "${remote_output}" "${PROJECT_ROOT}/${OUTPUT_REL}"
  fi

  log "Downloading benchmark log"
  scp_from_remote "${host}" "${port}" "${remote_log}" "${PROJECT_ROOT}/benchmarks/benchmark_vast_latest.log"

  if [[ "${exit_code}" != "0" ]]; then
    die "Remote benchmark exited with status ${exit_code}. See benchmarks/benchmark_vast_latest.log"
  fi
}

IMAGE="${DEFAULT_IMAGE}"
MODEL="${DEFAULT_MODEL}"
BENCHMARKS="${DEFAULT_BENCHMARKS}"
OUTPUT_REL="${DEFAULT_OUTPUT}"
REMOTE_DIR="${DEFAULT_REMOTE_DIR}"
DISK_GB="${DEFAULT_DISK_GB}"
BOOT_TIMEOUT="${DEFAULT_BOOT_TIMEOUT}"
POLL_SECONDS="${DEFAULT_POLL_SECONDS}"
MAX_OFFERS_PER_QUERY="${DEFAULT_MAX_OFFERS_PER_QUERY}"
WAIT_FOR_COMPLETION="0"
DESTROY_ON_SUCCESS="0"
KEEP_FAILED_INSTANCE="0"
OFFER_ID=""
READY_HOST=""
READY_PORT=""

VAST_BIN="$(resolve_vast_bin)"
SSH_PRIVATE_KEY="${SSH_PRIVATE_KEY:-${HOME}/.ssh/id_rsa}"
SSH_PUBLIC_KEY="${SSH_PUBLIC_KEY:-${SSH_PRIVATE_KEY}.pub}"
WEIGHTS_INPUT="${DEFAULT_WEIGHTS}"
HUMANEVAL_INPUT=""

declare -a OFFER_QUERIES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offer-id)
      OFFER_ID="${2:?missing offer id}"
      shift 2
      ;;
    --query)
      OFFER_QUERIES+=("${2:?missing query}")
      shift 2
      ;;
    --image)
      IMAGE="${2:?missing image}"
      shift 2
      ;;
    --disk)
      DISK_GB="${2:?missing disk size}"
      shift 2
      ;;
    --model)
      MODEL="${2:?missing model}"
      shift 2
      ;;
    --weights)
      WEIGHTS_INPUT="${2:?missing weights path}"
      shift 2
      ;;
    --humaneval-path)
      HUMANEVAL_INPUT="${2:?missing humaneval path}"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS="${2:?missing benchmark list}"
      shift 2
      ;;
    --output)
      OUTPUT_REL="${2:?missing output path}"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="${2:?missing remote dir}"
      shift 2
      ;;
    --boot-timeout)
      BOOT_TIMEOUT="${2:?missing timeout}"
      shift 2
      ;;
    --wait)
      WAIT_FOR_COMPLETION="1"
      shift
      ;;
    --destroy-on-success)
      DESTROY_ON_SUCCESS="1"
      shift
      ;;
    --keep-failed-instance)
      KEEP_FAILED_INSTANCE="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -x "${VAST_BIN}" ]] || die "vastai executable not found: ${VAST_BIN}"
ensure_file_exists "${SSH_PRIVATE_KEY}"
ensure_file_exists "${SSH_PUBLIC_KEY}"

if [[ "${WEIGHTS_INPUT}" = /* ]]; then
  WEIGHTS_PATH="${WEIGHTS_INPUT}"
else
  WEIGHTS_PATH="${PROJECT_ROOT}/${WEIGHTS_INPUT}"
fi
ensure_file_exists "${WEIGHTS_PATH}"
WEIGHTS_REL="$(make_relative_to_repo "${WEIGHTS_PATH}")"

HUMANEVAL_PATH_REL=""
if contains_benchmark "humaneval"; then
  if [[ -n "${HUMANEVAL_INPUT}" ]]; then
    if [[ "${HUMANEVAL_INPUT}" = /* ]]; then
      HUMANEVAL_PATH="${HUMANEVAL_INPUT}"
    else
      HUMANEVAL_PATH="${PROJECT_ROOT}/${HUMANEVAL_INPUT}"
    fi
  elif [[ -f "${PROJECT_ROOT}/benchmarks/results/HumanEval.jsonl" ]]; then
    HUMANEVAL_PATH="${PROJECT_ROOT}/benchmarks/results/HumanEval.jsonl"
  else
    die "HumanEval.jsonl is required for humaneval runs."
  fi
  ensure_file_exists "${HUMANEVAL_PATH}"
  HUMANEVAL_PATH_REL="$(make_relative_to_repo "${HUMANEVAL_PATH}")"
fi

BUNDLE_PATH="${TMPDIR:-/tmp}/ncpu_vast_coprocessor_bundle.tar.gz"
build_bundle "${BUNDLE_PATH}"

if [[ ${#OFFER_QUERIES[@]} -eq 0 && -z "${OFFER_ID}" ]]; then
  OFFER_QUERIES=(
    "gpu_name=RTX_4090 reliability > 0.97 num_gpus=1 dph_total<0.60"
    "gpu_name=RTX_3090 reliability > 0.97 num_gpus=1 dph_total<0.60"
    "gpu_name=A100 reliability > 0.98 num_gpus=1 dph_total<1.50"
  )
fi

declare -a CANDIDATE_OFFERS=()

if [[ -n "${OFFER_ID}" ]]; then
  CANDIDATE_OFFERS=("${OFFER_ID}")
else
  log "Searching Vast offers"
  declare -A seen_offers=()
  local_query=""
  for local_query in "${OFFER_QUERIES[@]}"; do
    while IFS= read -r found_offer; do
      [[ -n "${found_offer}" ]] || continue
      if [[ -z "${seen_offers[${found_offer}]:-}" ]]; then
        seen_offers["${found_offer}"]=1
        CANDIDATE_OFFERS+=("${found_offer}")
      fi
    done < <(offer_ids_from_query "${local_query}")
  done
fi

[[ ${#CANDIDATE_OFFERS[@]} -gt 0 ]] || die "No Vast offers matched the query set."
log "Offer candidates: ${CANDIDATE_OFFERS[*]}"

INSTANCE_ID=""
for candidate_offer in "${CANDIDATE_OFFERS[@]}"; do
  if ! INSTANCE_ID="$(create_instance_from_offer "${candidate_offer}")"; then
    log "Offer ${candidate_offer} failed to create"
    continue
  fi

  if wait_for_ssh_ready "${INSTANCE_ID}"; then
    break
  fi

  log "Instance ${INSTANCE_ID} never reached SSH readiness"
  if [[ "${KEEP_FAILED_INSTANCE}" != "1" ]]; then
    destroy_instance "${INSTANCE_ID}"
  fi
  INSTANCE_ID=""
done

[[ -n "${INSTANCE_ID}" ]] || die "No candidate reached SSH readiness."

log "Instance ready: ${INSTANCE_ID} (${READY_HOST}:${READY_PORT})"
REMOTE_LOG_PATH="$(start_remote_benchmark "${READY_HOST}" "${READY_PORT}")"

printf '\n'
printf 'Instance ID: %s\n' "${INSTANCE_ID}"
printf 'SSH: ssh -F /dev/null -i %q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p %s root@%s\n' \
  "${SSH_PRIVATE_KEY}" "${READY_PORT}" "${READY_HOST}"
printf 'Remote log: %s\n' "${REMOTE_LOG_PATH}"
printf 'Tail log: ssh -F /dev/null -i %q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p %s root@%s tail -f %q\n' \
  "${SSH_PRIVATE_KEY}" "${READY_PORT}" "${READY_HOST}" "${REMOTE_LOG_PATH}"
printf '\n'

if [[ "${WAIT_FOR_COMPLETION}" == "1" ]]; then
  wait_for_remote_completion "${READY_HOST}" "${READY_PORT}" "${REMOTE_LOG_PATH}"
  log "Local output saved to ${OUTPUT_REL}"

  if [[ "${DESTROY_ON_SUCCESS}" == "1" ]]; then
    destroy_instance "${INSTANCE_ID}"
  fi
fi

#!/usr/bin/env bash
set -euo pipefail

LOCAL_BENCH_DIR="/Users/bobbyprice/projects/nCPU/benchmarks"
LOCAL_TRAJECTORY_DIR="$LOCAL_BENCH_DIR/internal_trajectories"

PRIMARY_INSTANCE_ID="32666214"
SECONDARY_INSTANCE_ID="32678361"
PRIMARY_REMOTE_DIR="/root/queued_benchmarks"
SECONDARY_REMOTE_DIR="/root/queued_benchmarks"

mkdir -p "$LOCAL_BENCH_DIR"
mkdir -p "$LOCAL_TRAJECTORY_DIR"

resolve_instance_ssh() {
  local instance_id="$1"
  local ssh_url
  ssh_url="$(vastai ssh-url "$instance_id" 2>/dev/null | tail -n 1)" || return 1
  [[ -n "$ssh_url" ]] || return 1
  echo "$ssh_url" | sed -E 's#ssh://([^@]+)@([^:]+):([0-9]+)#\1 \2 \3#'
}

sync_file_from_instance() {
  local instance_id="$1"
  local remote_path="$2"
  local local_path="$3"
  local user host port

  read -r user host port < <(resolve_instance_ssh "$instance_id") || return 0
  mkdir -p "$(dirname "$local_path")"
  scp -q -o StrictHostKeyChecking=no -P "$port" \
    "${user}@${host}:${remote_path}" \
    "$local_path" 2>/dev/null || true
}

sync_tree_from_instance() {
  local instance_id="$1"
  local remote_dir="$2"
  local local_dir="$3"
  local user host port

  read -r user host port < <(resolve_instance_ssh "$instance_id") || return 0
  mkdir -p "$local_dir"

  if ssh -o StrictHostKeyChecking=no -p "$port" "${user}@${host}" "command -v rsync >/dev/null 2>&1 && test -d '$remote_dir'" >/dev/null 2>&1; then
    rsync -az \
      -e "ssh -o StrictHostKeyChecking=no -p $port" \
      "${user}@${host}:${remote_dir}/" \
      "$local_dir/" >/dev/null 2>&1 || true
    return
  fi

  if ssh -o StrictHostKeyChecking=no -p "$port" "${user}@${host}" "test -d '$remote_dir'" >/dev/null 2>&1; then
    local archive
    archive="$(mktemp -t sync_parallel_benchmarks.XXXXXX.tar)"
    if ssh -o StrictHostKeyChecking=no -p "$port" "${user}@${host}" "cd '$remote_dir' && tar -cf - ." >"$archive" 2>/dev/null; then
      tar -xf "$archive" -C "$local_dir" >/dev/null 2>&1 || true
    fi
    rm -f "$archive"
  fi
}

sync_named_outputs() {
  local instance_id="$1"
  local remote_dir="$2"
  shift 2

  local name
  for name in "$@"; do
    sync_file_from_instance "$instance_id" "$remote_dir/$name" "$LOCAL_BENCH_DIR/$name"
  done
}

sync_from_primary() {
  sync_named_outputs \
    "$PRIMARY_INSTANCE_ID" \
    "$PRIMARY_REMOTE_DIR" \
    "queue_single_h200_benchmarks.log" \
    "bigcodebench_hard_instruct_qwen35_4b_some.log" \
    "bigcodebench_hard_instruct_qwen35_4b_some.json" \
    "bigcodebench_hard_instruct_qwen35_4b_some.json.progress.jsonl" \
    "bigcodebench_hard_instruct_qwen35_4b_resume.log" \
    "evalplus_mbpp_full_qwen35_4b_some.log" \
    "evalplus_mbpp_full_qwen35_4b_some.json" \
    "evalplus_mbpp_full_qwen35_4b_some.json.progress.jsonl"

  sync_tree_from_instance \
    "$PRIMARY_INSTANCE_ID" \
    "$PRIMARY_REMOTE_DIR/internal_trajectories" \
    "$LOCAL_TRAJECTORY_DIR"
}

sync_from_secondary() {
  sync_named_outputs \
    "$SECONDARY_INSTANCE_ID" \
    "$SECONDARY_REMOTE_DIR" \
    "queue_parallel_h200_benchmarks.log" \
    "bigcodebench_hard_instruct_qwen35_9b_some.log" \
    "bigcodebench_hard_instruct_qwen35_9b_some.json" \
    "bigcodebench_hard_instruct_qwen35_9b_some.json.progress.jsonl" \
    "bigcodebench_hard_instruct_qwen35_9b_resume.log" \
    "bigcodebench_hard_instruct_qwen35_27b_some.log" \
    "bigcodebench_hard_instruct_qwen35_27b_some.json" \
    "bigcodebench_hard_instruct_qwen35_27b_some.json.progress.jsonl" \
    "bigcodebench_hard_instruct_qwen35_27b_resume.log" \
    "evalplus_mbpp_full_qwen35_9b_some.log" \
    "evalplus_mbpp_full_qwen35_9b_some.json" \
    "evalplus_mbpp_full_qwen35_9b_some.json.progress.jsonl" \
    "evalplus_mbpp_full_qwen35_27b_some.log" \
    "evalplus_mbpp_full_qwen35_27b_some.json" \
    "evalplus_mbpp_full_qwen35_27b_some.json.progress.jsonl"

  sync_tree_from_instance \
    "$SECONDARY_INSTANCE_ID" \
    "$SECONDARY_REMOTE_DIR/internal_trajectories" \
    "$LOCAL_TRAJECTORY_DIR"
}

copy_finished_secondary_outputs_to_primary() {
  local user host port
  read -r user host port < <(resolve_instance_ssh "$PRIMARY_INSTANCE_ID") || return 0

  local names=(
    "bigcodebench_hard_instruct_qwen35_9b_some.json"
    "bigcodebench_hard_instruct_qwen35_27b_some.json"
    "evalplus_mbpp_full_qwen35_9b_some.json"
    "evalplus_mbpp_full_qwen35_27b_some.json"
  )

  local name
  for name in "${names[@]}"; do
    if [[ -f "$LOCAL_BENCH_DIR/$name" ]]; then
      scp -q -o StrictHostKeyChecking=no -P "$port" \
        "$LOCAL_BENCH_DIR/$name" \
        "${user}@${host}:${PRIMARY_REMOTE_DIR}/$name" 2>/dev/null || true
    fi
  done
}

main() {
  while true; do
    sync_from_primary
    sync_from_secondary
    copy_finished_secondary_outputs_to_primary
    sleep 60
  done
}

main "$@"

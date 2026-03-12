#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root"
QUEUE_DIR="/root/queued_benchmarks"
TRAJECTORY_ROOT="$QUEUE_DIR/internal_trajectories"
BASE_URL="http://127.0.0.1:11434"
REQUEST_TIMEOUT="300"
MAX_RETRIES="3"

mkdir -p "$QUEUE_DIR"
mkdir -p "$TRAJECTORY_ROOT"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

LOCK_DIR="$QUEUE_DIR/.queue_parallel.lock"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

ensure_ollama_running() {
  if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
    log "starting ollama serve"
    nohup ollama serve >/root/ollama.log 2>&1 </dev/null &
    sleep 3
  fi
}

ensure_model() {
  local model="$1"
  if ollama list | awk 'NR > 1 {print $1}' | grep -qx "$model"; then
    log "model present: $model"
    return
  fi
  log "pulling model: $model"
  ollama pull "$model"
}

ensure_bigcodebench() {
  if python3 - <<'PY' >/dev/null 2>&1
from bigcodebench.data import get_bigcodebench
from bigcodebench.eval import PASS, untrusted_check
from bigcodebench.gen.util import trusted_check
from bigcodebench.sanitize import sanitize
import matplotlib.pyplot  # noqa: F401
PY
  then
    log "BigCodeBench dependency already installed"
    return
  fi

  log "installing BigCodeBench dependency set"
  python3 -m pip install --no-deps \
    bigcodebench \
    appdirs \
    bounded-pool-executor \
    tempdir \
    wget \
    pqdm \
    termcolor \
    tqdm \
    tree_sitter \
    tree-sitter-python \
    datasets \
    huggingface-hub \
    matplotlib \
    multiprocess \
    numpy \
    pyarrow \
    rich \
    xxhash
}

ensure_bigcodebench_task_runtime() {
  if python3 - <<'PY' >/dev/null 2>&1
from flask import Flask
from flask_login import LoginManager
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from wordcloud import WordCloud
PY
  then
    log "BigCodeBench task runtime already installed"
    return
  fi

  log "installing BigCodeBench task runtime packages"
  python3 -m pip install \
    Flask \
    Flask-Login \
    Flask-Mail \
    Flask-RESTful \
    Flask-WTF \
    WTForms \
    Faker \
    beautifulsoup4 \
    blake3 \
    chardet \
    django \
    folium \
    geopandas \
    geopy \
    gensim \
    holidays \
    librosa \
    lxml \
    mechanize \
    natsort \
    nltk \
    opencv-python-headless \
    openpyxl \
    prettytable \
    pycryptodome \
    pyquery \
    pytesseract \
    python-Levenshtein \
    python-dateutil \
    python-docx \
    scikit-image \
    scikit-learn \
    scipy \
    seaborn \
    sendgrid \
    shapely \
    soundfile \
    statsmodels \
    tensorflow \
    textblob \
    texttable \
    wikipedia \
    wordcloud \
    wordninja \
    xlwt \
    xmltodict
}

run_step() {
  local name="$1"
  local output_path="$2"
  local log_path="$3"
  shift 3

  if [[ -f "$output_path" ]]; then
    log "skip $name: output already exists at $output_path"
    return
  fi

  log "start $name"
  "$@" | tee "$log_path"
  log "finish $name"
}

run_bigcodebench_step() {
  local name="$1"
  local model="$2"
  local output_path="$3"
  local log_path="$4"
  local progress_path="${output_path}.progress.jsonl"

  if [[ -f "$output_path" ]]; then
    log "skip $name: output already exists at $output_path"
    return
  fi

  log "start $name"
  if [[ -f "$progress_path" ]]; then
    log "resume $name from checkpoint $progress_path"
    python3 -u ncpu/self_optimizing/resume_bigcodebench_benchmark.py \
      --provider local \
      --subset hard \
      --split instruct \
      --model "$model" \
      --progress-path "$progress_path" \
      --trajectory-root "$TRAJECTORY_ROOT/bigcodebench/hard_instruct" \
      --output "$output_path" | tee "$log_path"
  else
    python3 -u ncpu/self_optimizing/run_bigcodebench_benchmark.py \
      --provider local \
      --subset hard \
      --split instruct \
      --model "$model" \
      --base-url "$BASE_URL" \
      --max-retries "$MAX_RETRIES" \
      --groundtruth-workers 4 \
      --trajectory-root "$TRAJECTORY_ROOT/bigcodebench/hard_instruct" \
      --output "$output_path" | tee "$log_path"
  fi
  log "finish $name"
}

main() {
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "queue already running: $LOCK_DIR"
    exit 0
  fi
  trap 'rmdir "$LOCK_DIR" >/dev/null 2>&1 || true' EXIT

  ensure_ollama_running
  ensure_bigcodebench
  ensure_bigcodebench_task_runtime
  ensure_model "qwen3.5:9b"
  ensure_model "qwen3.5:27b"

  run_bigcodebench_step \
    "bigcodebench-hard-9b" \
    "qwen3.5:9b" \
    "$QUEUE_DIR/bigcodebench_hard_instruct_qwen35_9b_some.json" \
    "$QUEUE_DIR/bigcodebench_hard_instruct_qwen35_9b_some.log"

  run_bigcodebench_step \
    "bigcodebench-hard-27b" \
    "qwen3.5:27b" \
    "$QUEUE_DIR/bigcodebench_hard_instruct_qwen35_27b_some.json" \
    "$QUEUE_DIR/bigcodebench_hard_instruct_qwen35_27b_some.log"

  run_step \
    "mbpp-full-9b" \
    "$QUEUE_DIR/evalplus_mbpp_full_qwen35_9b_some.json" \
    "$QUEUE_DIR/evalplus_mbpp_full_qwen35_9b_some.log" \
    python3 -u ncpu/self_optimizing/run_evalplus_benchmark.py \
      --provider local \
      --dataset mbpp \
      --full \
      --model qwen3.5:9b \
      --base-url "$BASE_URL" \
      --request-timeout "$REQUEST_TIMEOUT" \
      --max-retries "$MAX_RETRIES" \
      --trajectory-root "$TRAJECTORY_ROOT/evalplus/mbpp" \
      --output "$QUEUE_DIR/evalplus_mbpp_full_qwen35_9b_some.json"

  run_step \
    "mbpp-full-27b" \
    "$QUEUE_DIR/evalplus_mbpp_full_qwen35_27b_some.json" \
    "$QUEUE_DIR/evalplus_mbpp_full_qwen35_27b_some.log" \
    python3 -u ncpu/self_optimizing/run_evalplus_benchmark.py \
      --provider local \
      --dataset mbpp \
      --full \
      --model qwen3.5:27b \
      --base-url "$BASE_URL" \
      --request-timeout "$REQUEST_TIMEOUT" \
      --max-retries "$MAX_RETRIES" \
      --trajectory-root "$TRAJECTORY_ROOT/evalplus/mbpp" \
      --output "$QUEUE_DIR/evalplus_mbpp_full_qwen35_27b_some.json"

  log "parallel queue complete"
}

main "$@"

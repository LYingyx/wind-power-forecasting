#!/usr/bin/env bash

# Source this file from one launcher at a time.  Completion labels written to
# the coprocess stdin become updates on a persistent terminal tqdm bar.
paper_tqdm_start() {
    local total=$1
    local description=$2
    local unit=${3:-job}
    coproc PAPER_TQDM_PROCESS {
        "$PYTHON_BIN" "$PROJECT_ROOT/scripts/tqdm_job_progress.py" \
            --total "$total" --desc "$description" --unit "$unit" --position 0
    }
    PAPER_TQDM_PID_SAVED=$PAPER_TQDM_PROCESS_PID
    PAPER_TQDM_WRITE_FD=${PAPER_TQDM_PROCESS[1]}
}

paper_tqdm_update() {
    printf '%s\n' "$1" >&"$PAPER_TQDM_WRITE_FD"
}

paper_tqdm_finish() {
    if [[ -n "${PAPER_TQDM_WRITE_FD:-}" ]]; then
        exec {PAPER_TQDM_WRITE_FD}>&-
    fi
    if [[ -n "${PAPER_TQDM_PID_SAVED:-}" ]]; then
        wait "$PAPER_TQDM_PID_SAVED" 2>/dev/null || true
    fi
    PAPER_TQDM_WRITE_FD=
    PAPER_TQDM_PID_SAVED=
}

paper_tqdm_abort() {
    if [[ -n "${PAPER_TQDM_PID_SAVED:-}" ]]; then
        kill -TERM "$PAPER_TQDM_PID_SAVED" 2>/dev/null || true
        wait "$PAPER_TQDM_PID_SAVED" 2>/dev/null || true
    fi
    PAPER_TQDM_WRITE_FD=
    PAPER_TQDM_PID_SAVED=
}

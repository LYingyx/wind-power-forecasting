# Complete paper experiment suite

## Formal launch

Run the complete deterministic experiment matrix in one tmux window:

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
bash shell/launch_paper_suite.sh 0,1,2,3,4,5,6,7
```

The comma-separated values are physical GPU indices. The launcher creates one
worker per listed GPU, prints prefixed tqdm progress bars and key metrics in the
current window, waits for each dependency stage, and exits only after analysis
has finished. It does not use `nohup` or create additional tmux windows.

The formal suite uses seeds 42, 43, and 44. Override the interpreter, config,
or output directory only when required:

```bash
CHRONOS_PYTHON=/data/liying/anaconda3/envs/chronos/bin/python \
WIND_BASE_CONFIG=/storage/liying/new_try/wind_cluster_forced_evolution/configs/forced_evolution_pad5.yaml \
WIND_PAPER_ROOT=/storage/liying/new_try/wind_cluster_forced_evolution/artifacts/paper_experiments_pad5_v1 \
bash shell/launch_paper_suite.sh 0,1,2,3,4,5,6,7
```

Do not set `WIND_SMOKE=1` for formal experiments. That option deliberately
limits each epoch and evaluation to one batch and is only an integration test.

## Experiment coverage

The suite trains and validates 165 runs:

| Stage | Runs | Purpose |
|---|---:|---|
| Spatial local support | 15 | nearest point, fixed bilinear cell, learned 1/3/5-cell support |
| Spatial geometry | 6 | wind geometry and regional context increments |
| Forcing path | 12 | independent, difference, TCN, and Transformer hourly paths |
| Time alignment | 12 | nearest, bracketing MLP/attention, and full-path attention |
| Source fusion | 18 | history/weather contribution and proposed fusion ablations |
| Output hierarchy | 9 | direct cluster, station-only, and joint hierarchy |
| Training controls | 12 | spatial pretraining and auxiliary-loss 2 by 2 control |
| History baselines | 30 | persistence, Linear, DLinear, LSTM, TCN, Transformer, PatchTST, TFT, iTransformer |
| Weather baselines | 30 | point/cell MLP, ResNet, 3D CNN, ConvLSTM, ViT, Swin, Video Swin, hierarchical controls |
| Fusion baselines | 21 | prediction weighting, concatenation, gating, cross-attention, TFT, ASTAFN adaptation |

After validation-only selection is frozen, the launcher performs 84 test
evaluations: the three-seed final model and every history, weather, and fusion
baseline. The main path uses native grid values and learned queries; fixed
bilinear mapping is retained only as a diagnostic baseline.

The spatial terminology counts grid cells. A learned 1 by 1, 3 by 3, or 5 by 5
cell neighborhood contains 2 by 2, 4 by 4, or 6 by 6 native grid points.
`s0_nearest_point` reads one point, while `s0b_bilinear_cell1` reads the four
corners of the enclosing cell.

## Scheduling and selection

Stages run in dependency order. Within a stage, up to one job per supplied GPU
runs concurrently. Each module choice is based on the mean validation cluster
MAE across all three seeds. Test predictions are not read until every choice
has been frozen.

The final comparison reports MAE, RMSE, nMAE, R-squared, persistence skill,
6-hour energy error, per-lead errors, 0-1/1-3/3-6-hour errors, parameter count,
and inference throughput. Three-seed ensemble predictions are compared with
baselines by paired UTC-day block bootstrap. The rapid-change result is an
evaluation diagnostic only; it is not a training objective or primary
scientific question.

## Outputs and restart

The default root is:

```text
artifacts/paper_experiments_pad5_v1/
  spatial/ forcing/ time_alignment/ fusion/ hierarchy/ controls/
  baselines_history/ baselines_weather/ baselines_fusion/
  selected/ final/ logs/ summary/ analysis/
  ALL_COMPLETE
```

Each completed forecast training run contains `config.yaml`, `best_model.pt`,
`history.json`, `metrics_val.json`, `predictions_val.npz`, `run.log`, and
`COMPLETE`. Spatial-pretraining runs contain the same files except the forecast
prediction archive. Frozen test runs additionally contain `metrics_test.json`,
`predictions_test.npz`, and `TEST_COMPLETE`.

Important aggregate outputs are:

- `summary/summary.csv` and `summary/summary.json`;
- `analysis/runs.csv` and `analysis/groups.csv`;
- `analysis/paired_block_bootstrap.csv`;
- `analysis/rapid_change_diagnostic.json`.

Rerunning the same launch command skips runs with completion markers and
continues incomplete work. A conflicting saved configuration raises an error
instead of silently mixing results. Use a new `WIND_PAPER_ROOT` after changing
model or training settings. `Ctrl-C` terminates the active child jobs started
by the launcher.

Monitor progress from another terminal when needed:

```bash
find artifacts/paper_experiments_pad5_v1 -name COMPLETE -o -name TEST_COMPLETE
cat artifacts/paper_experiments_pad5_v1/logs/paper_jobs.tsv
```

## Integration smoke test

The following command validates every stage with one train/validation/test
batch and writes outside the formal result directory:

```bash
WIND_SMOKE=1 \
WIND_PAPER_ROOT=/tmp/wind_paper_suite_smoke \
bash shell/launch_paper_suite.sh 0,1,2
```

## Controlled-transition follow-up

The controlled-transition suite waits for the formal suite to freeze its spatial,
forcing-path, and time-alignment choices. It then compares the existing parallel
decoder with controlled GRU, stable selective SSM, controlled Koopman, and a
multi-rate selective SSM. Every candidate uses seeds 42, 43, and 44; only the
validation-selected family is evaluated on the test split.

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
bash shell/launch_transition_suite.sh 2,3,4,6,7
```

Use `WIND_TRANSITION_SOURCE_CONFIG` only for an immediate pilot that deliberately
bypasses the formal validation-frozen upstream choices. The following exercises
the complete scheduler with one batch per train/validation/test phase:

```bash
WIND_SMOKE=1 \
WIND_TRANSITION_SOURCE_CONFIG=configs/controlled_transition_pad5.yaml \
WIND_TRANSITION_ROOT=/tmp/wind_transition_suite_smoke \
bash shell/launch_transition_suite.sh 0,1,2
```

## Final route confirmation

Both completed suites leave one unresolved fair comparison: the validation-selected
parallel decoder used an hourly station-weather auxiliary loss, while the controlled
GRU screen did not. The final-route suite trains the controlled GRU with the same
auxiliary weight and full selected upstream configuration. It also runs four focused
ablations for wind-direction geometry, learned station mapping, hourly weather
changes, and access to the full future weather path.

Only the formal parallel model and the new complete controlled model participate in
validation selection. Ablations are evidence for individual design choices and are
never eligible to become the main model. Test data are evaluated only after this
selection is frozen.

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
bash shell/launch_final_route_suite.sh 5,6,7
```

The launcher schedules 15 training jobs over the supplied GPUs, then evaluates the
three selected checkpoints on test. It is restartable: completed runs are skipped,
while conflicting saved configurations stop with an error. Outputs are written to
`artifacts/final_route_pad5_v1`, including validation-only UTC-day bootstrap
comparisons under `analysis/`.

Integration smoke test:

```bash
WIND_SMOKE=1 \
WIND_FINAL_ROUTE_ROOT=/tmp/wind_final_route_suite_smoke \
bash shell/launch_final_route_suite.sh 5,6,7
```

## Official Mamba state-transition baselines

The local `selective_ssm` transition is a small diagonal controlled state-space
model; it is not Mamba. Official Mamba and Mamba2 are provided as separate causal
state-transition baselines through the pinned `mamba-ssm==2.2.6.post3` package.
They receive the same history-initialized station state and quarter-hour weather
controls as the controlled GRU. The initial state is the first sequence token, so
the official causal scan carries it into the following 24 forecast tokens.

Install the prebuilt wheel without changing the existing Torch environment. The
default URL is pinned for Python 3.11, CUDA 12, Torch 2.11, and CXX11 ABI builds;
a downloaded local wheel can be supplied as the first argument when GitHub is not
reachable:

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
bash shell/install_mamba_ssm.sh /path/to/mamba_ssm-2.2.6.post3+cu12torch2.11cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

The Mamba suite is independent from an already running final-route suite. It exits
before scheduling if the official package is absent and waits for
`artifacts/final_route_pad5_v1/ALL_COMPLETE` before using any GPUs. It trains and
tests Mamba and Mamba2 with three seeds, then compares them with the controlled GRU,
the parallel decoder, and the validation-selected final model.

```bash
bash shell/launch_mamba_baseline_suite.sh 5,6,7
```

Outputs are written to `artifacts/mamba_baselines_pad5_v1`. Repeating the same
command resumes incomplete work and skips completed runs.

## Multiscale controlled-state follow-up

The next validation-only optimization adds a station-conditioned coarse-grid
weather branch and quarter-hour power-change supervision. It first screens
spatial scales, change-loss weight, state width, and dropout with seed 42. It
then runs three-seed confirmation for the complete controlled-state model, a
matched parallel decoder, and removal controls for both new components.

The state model is not forced to win. Mean validation MAE selects either the
complete state family or the matched parallel family before three frozen test
evaluations. All new training outputs are isolated under
`artifacts/multiscale_state_pad5_v1`.

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
bash shell/launch_multiscale_state_suite.sh 3,4,5,6,7
```

The launcher is restartable and schedules one process per supplied physical
GPU. A one-batch integration run uses a separate output directory:

```bash
WIND_SMOKE=1 \
WIND_MULTISCALE_STATE_ROOT=/tmp/wind_multiscale_state_smoke \
bash shell/launch_multiscale_state_suite.sh 3,4,5
```

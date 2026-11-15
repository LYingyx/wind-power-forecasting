#!/usr/bin/env bash
set -euo pipefail

SHELL_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SHELL_DIR/.." && pwd)
PYTHON_BIN=${CHRONOS_PYTHON:-/data/liying/anaconda3/envs/chronos/bin/python}
BASE_CONFIG=${WIND_BASE_CONFIG:-$PROJECT_ROOT/configs/forced_evolution_pad5.yaml}
EXPERIMENT_ROOT=${WIND_PAPER_ROOT:-$PROJECT_ROOT/artifacts/paper_experiments_pad5_v1}
GPU_CSV=${1:-0,1,2,3,4,5,6,7}
SEED_CSV=${WIND_PAPER_SEEDS:-42,43,44}
IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a SEEDS <<< "$SEED_CSV"

if [[ ${#GPUS[@]} -lt 1 ]]; then
    echo "At least one GPU index is required" >&2
    exit 2
fi
if [[ ${#SEEDS[@]} -ne 3 ]]; then
    echo "WIND_PAPER_SEEDS must contain exactly three seeds" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python interpreter: $PYTHON_BIN" >&2
    exit 2
fi

mkdir -p "$EXPERIMENT_ROOT/logs" "$EXPERIMENT_ROOT/selected"
JOBS_FILE="$EXPERIMENT_ROOT/logs/paper_jobs.tsv"
printf 'stage\tindex\tgpu\tpid\tjob\n' > "$JOBS_FILE"
export WIND_EXPERIMENT_ROOT="$EXPERIMENT_ROOT"

ACTIVE_PIDS=()
stop_active() {
    trap - INT TERM
    echo
    echo "[paper] stopping ${#ACTIVE_PIDS[@]} active jobs"
    for pid in "${ACTIVE_PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
}
trap stop_active INT TERM

JOBS=()
add_job() {
    local IFS='|'
    JOBS+=("$*")
}

run_stage() {
    local stage=$1
    shift
    local jobs=("$@")
    local total=${#jobs[@]}
    local start=0
    echo "[paper] stage-start name=$stage jobs=$total"
    while (( start < total )); do
        ACTIVE_PIDS=()
        local labels=()
        local slots=${#GPUS[@]}
        local end=$((start + slots))
        if (( end > total )); then
            end=$total
        fi
        local index
        for ((index=start; index<end; index++)); do
            local gpu=${GPUS[$((index - start))]}
            local fields=()
            IFS='|' read -r -a fields <<< "${jobs[$index]}"
            local prefix="[$stage:$index]"
            WIND_LANE_ID="$stage-$index" \
            WIND_PROGRESS_PREFIX="$prefix" \
            WIND_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
            bash "$SHELL_DIR/run_paper_job.sh" "$gpu" "${fields[@]}" &
            local pid=$!
            ACTIVE_PIDS+=("$pid")
            labels+=("${jobs[$index]}")
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$stage" "$index" "$gpu" "$pid" "${jobs[$index]}" >> "$JOBS_FILE"
            echo "[paper] started stage=$stage index=$index gpu=$gpu pid=$pid"
        done
        local failed=0
        local offset
        for offset in "${!ACTIVE_PIDS[@]}"; do
            local pid=${ACTIVE_PIDS[$offset]}
            if wait "$pid"; then
                echo "[paper] finished stage=$stage pid=$pid status=0"
            else
                local status=$?
                echo "[paper] failed stage=$stage pid=$pid status=$status job=${labels[$offset]}" >&2
                failed=1
            fi
        done
        ACTIVE_PIDS=()
        if (( failed != 0 )); then
            return 1
        fi
        start=$end
    done
    echo "[paper] stage-complete name=$stage"
}

select_group() {
    local output=$1
    shift
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/select_best_group.py" \
        --output-dir "$output" \
        --metric cluster_mae_mw \
        --seeds "$SEED_CSV" \
        "$@"
}

common_forced_overrides=(
    model.architecture=forced_evolution
    model.output_level=hierarchical
    training.deterministic.station_loss_weight=1.0
    training.deterministic.cluster_loss_weight=2.0
    training.deterministic.auxiliary_weather_loss_weight=0.0
)

# S0-S3: fixed spatial mappings and learned local support.
JOBS=()
for seed in "${SEEDS[@]}"; do
    add_job spatial spatial "s0_nearest_point/seed$seed" "$BASE_CONFIG" NONE \
        "seed=$seed" model.station_query_mode=nearest \
        model.station_neighborhood_cells=1 model.include_wind_geometry=false \
        model.include_region_context=false model.forcing_path_mode=independent
    add_job spatial spatial "s0b_bilinear_cell1/seed$seed" "$BASE_CONFIG" NONE \
        "seed=$seed" model.station_query_mode=bilinear \
        model.station_neighborhood_cells=1 model.include_wind_geometry=false \
        model.include_region_context=false model.forcing_path_mode=independent
    for cells in 1 3 5; do
        case "$cells" in
            1) spatial_id=s1_learned_cell1 ;;
            3) spatial_id=s2_learned_cell3 ;;
            5) spatial_id=s3_learned_cell5 ;;
        esac
        add_job spatial spatial "$spatial_id/seed$seed" "$BASE_CONFIG" NONE \
            "seed=$seed" model.station_query_mode=learned_cells \
            "model.station_neighborhood_cells=$cells" \
            model.include_wind_geometry=false model.include_region_context=false \
            model.forcing_path_mode=independent
    done
done
run_stage spatial_local "${JOBS[@]}"
select_group "$EXPERIMENT_ROOT/selected/local" \
    "$EXPERIMENT_ROOT/spatial/s1_learned_cell1" \
    "$EXPERIMENT_ROOT/spatial/s2_learned_cell3" \
    "$EXPERIMENT_ROOT/spatial/s3_learned_cell5"

# S4-S5: wind geometry and regional context increments.
JOBS=()
for seed in "${SEEDS[@]}"; do
    selected_local="$EXPERIMENT_ROOT/selected/local/seed$seed/config.yaml"
    add_job spatial spatial "s4_selected_local_wind/seed$seed" "$selected_local" NONE \
        "seed=$seed" model.include_wind_geometry=true \
        model.include_region_context=false model.forcing_path_mode=independent
    add_job spatial spatial "s5_selected_local_wind_context/seed$seed" "$selected_local" NONE \
        "seed=$seed" model.include_wind_geometry=true \
        model.include_region_context=true model.forcing_path_mode=independent
done
run_stage spatial_geometry "${JOBS[@]}"
select_group "$EXPERIMENT_ROOT/selected/spatial" \
    "$EXPERIMENT_ROOT/selected/local" \
    "$EXPERIMENT_ROOT/spatial/s4_selected_local_wind" \
    "$EXPERIMENT_ROOT/spatial/s5_selected_local_wind_context"

# F0-F3: hourly forcing-path encoders, trained from scratch without auxiliary loss.
JOBS=()
forcing_ids=(f0_independent f1_difference f2_tcn f3_transformer)
forcing_modes=(independent difference tcn transformer)
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/selected/spatial/seed$seed/config.yaml"
    for index in "${!forcing_ids[@]}"; do
        add_job forecast forcing "${forcing_ids[$index]}/seed$seed" "$base" NONE \
            "seed=$seed" "${common_forced_overrides[@]}" \
            "model.forcing_path_mode=${forcing_modes[$index]}" \
            model.time_query_mode=full_attention model.fusion_mode=state_conditioned
    done
done
run_stage forcing_path "${JOBS[@]}"
select_group "$EXPERIMENT_ROOT/selected/forcing" \
    "$EXPERIMENT_ROOT/forcing/f0_independent" \
    "$EXPERIMENT_ROOT/forcing/f1_difference" \
    "$EXPERIMENT_ROOT/forcing/f2_tcn" \
    "$EXPERIMENT_ROOT/forcing/f3_transformer"

# Q0-Q3: explicit 1-hour weather to 15-minute target-time response mappings.
JOBS=()
time_ids=(q0_nearest q1_bracketing_mlp q2_bracketing_attention q3_full_attention)
time_modes=(nearest bracketing_mlp bracketing_attention full_attention)
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/selected/forcing/seed$seed/config.yaml"
    for index in "${!time_ids[@]}"; do
        add_job forecast time_alignment "${time_ids[$index]}/seed$seed" "$base" NONE \
            "seed=$seed" "${common_forced_overrides[@]}" \
            "model.time_query_mode=${time_modes[$index]}" \
            model.fusion_mode=state_conditioned
    done
done
run_stage time_alignment "${JOBS[@]}"
select_group "$EXPERIMENT_ROOT/selected/time_alignment" \
    "$EXPERIMENT_ROOT/time_alignment/q0_nearest" \
    "$EXPERIMENT_ROOT/time_alignment/q1_bracketing_mlp" \
    "$EXPERIMENT_ROOT/time_alignment/q2_bracketing_attention" \
    "$EXPERIMENT_ROOT/time_alignment/q3_full_attention"

# T1-T6: source contribution and fusion mechanism ablations.
JOBS=()
fusion_ids=(t1_history_only t2_weather_only t3_direct_concat t4_target_concat t5_lead_gate t6_state_conditioned)
fusion_modes=(history_only weather_only direct_concat target_concat lead_gate state_conditioned)
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/selected/time_alignment/seed$seed/config.yaml"
    for index in "${!fusion_ids[@]}"; do
        add_job forecast fusion "${fusion_ids[$index]}/seed$seed" "$base" NONE \
            "seed=$seed" "${common_forced_overrides[@]}" \
            "model.fusion_mode=${fusion_modes[$index]}"
    done
done
run_stage source_fusion "${JOBS[@]}"
select_group "$EXPERIMENT_ROOT/selected/fusion" \
    "$EXPERIMENT_ROOT/fusion/t1_history_only" \
    "$EXPERIMENT_ROOT/fusion/t2_weather_only" \
    "$EXPERIMENT_ROOT/fusion/t3_direct_concat" \
    "$EXPERIMENT_ROOT/fusion/t4_target_concat" \
    "$EXPERIMENT_ROOT/fusion/t5_lead_gate" \
    "$EXPERIMENT_ROOT/fusion/t6_state_conditioned"

# H1-H3: direct cluster, station-only hierarchy, and joint hierarchy.
JOBS=()
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/selected/fusion/seed$seed/config.yaml"
    add_job forecast hierarchy "h1_direct_cluster/seed$seed" "$base" NONE \
        "seed=$seed" model.output_level=direct_cluster \
        training.deterministic.station_loss_weight=0.0 \
        training.deterministic.cluster_loss_weight=2.0 \
        training.deterministic.auxiliary_weather_loss_weight=0.0
    add_job forecast hierarchy "h2_station_only/seed$seed" "$base" NONE \
        "seed=$seed" model.output_level=hierarchical \
        training.deterministic.station_loss_weight=1.0 \
        training.deterministic.cluster_loss_weight=0.0 \
        training.deterministic.auxiliary_weather_loss_weight=0.0
    add_job forecast hierarchy "h3_joint_hierarchy/seed$seed" "$base" NONE \
        "seed=$seed" model.output_level=hierarchical \
        training.deterministic.station_loss_weight=1.0 \
        training.deterministic.cluster_loss_weight=2.0 \
        training.deterministic.auxiliary_weather_loss_weight=0.0
done
run_stage hierarchy "${JOBS[@]}"
select_group "$EXPERIMENT_ROOT/selected/hierarchy" \
    "$EXPERIMENT_ROOT/hierarchy/h1_direct_cluster" \
    "$EXPERIMENT_ROOT/hierarchy/h2_station_only" \
    "$EXPERIMENT_ROOT/hierarchy/h3_joint_hierarchy"

# 2x2 spatial-pretraining by hourly-auxiliary-loss control.
JOBS=()
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/selected/hierarchy/seed$seed/config.yaml"
    spatial_checkpoint="$EXPERIMENT_ROOT/selected/spatial/seed$seed/best_model.pt"
    add_job forecast controls "pretrained_aux/seed$seed" "$base" "$spatial_checkpoint" \
        "seed=$seed" \
        training.deterministic.auxiliary_weather_loss_weight=0.2
    add_job forecast controls "pretrained_noaux/seed$seed" "$base" "$spatial_checkpoint" \
        "seed=$seed" \
        training.deterministic.auxiliary_weather_loss_weight=0.0
    add_job forecast controls "scratch_aux/seed$seed" "$base" NONE \
        "seed=$seed" \
        training.deterministic.auxiliary_weather_loss_weight=0.2
    add_job forecast controls "scratch_noaux/seed$seed" "$base" NONE \
        "seed=$seed" \
        training.deterministic.auxiliary_weather_loss_weight=0.0
done
run_stage training_controls "${JOBS[@]}"
select_group "$EXPERIMENT_ROOT/selected/training" \
    "$EXPERIMENT_ROOT/controls/pretrained_aux" \
    "$EXPERIMENT_ROOT/controls/pretrained_noaux" \
    "$EXPERIMENT_ROOT/controls/scratch_aux" \
    "$EXPERIMENT_ROOT/controls/scratch_noaux"

# HP0-HP7 plus Linear and direct-cluster TCN controls.
JOBS=()
history_ids=(hp0_persistence hp1_linear hp1_dlinear hp2_lstm hp3_tcn hp3b_tcn_cluster hp4_transformer hp5_patchtst hp6_tft hp7_itransformer)
history_arch=(history_persistence history_linear history_dlinear history_lstm history_tcn history_tcn history_transformer history_patchtst history_tft history_itransformer)
for seed in "${SEEDS[@]}"; do
    for index in "${!history_ids[@]}"; do
        output_level=hierarchical
        station_weight=1.0
        epochs_override=training.deterministic.epochs=80
        if [[ ${history_ids[$index]} == hp3b_tcn_cluster ]]; then
            output_level=direct_cluster
            station_weight=0.0
        fi
        if [[ ${history_ids[$index]} == hp0_persistence ]]; then
            epochs_override=training.deterministic.epochs=1
        fi
        add_job forecast baselines_history "${history_ids[$index]}/seed$seed" "$BASE_CONFIG" NONE \
            "seed=$seed" "model.architecture=${history_arch[$index]}" \
            "model.output_level=$output_level" \
            "training.deterministic.station_loss_weight=$station_weight" \
            training.deterministic.cluster_loss_weight=2.0 \
            training.deterministic.auxiliary_weather_loss_weight=0.0 \
            "$epochs_override"
    done
done
run_stage history_baselines "${JOBS[@]}"
history_candidates=()
for id in "${history_ids[@]}"; do history_candidates+=("$EXPERIMENT_ROOT/baselines_history/$id"); done
select_group "$EXPERIMENT_ROOT/selected/history_baseline" "${history_candidates[@]}"

# WP0-WP7: direct-cluster weather encoders on native hourly grids.
JOBS=()
weather_ids=(wp0_nearest_mlp wp0b_bilinear_mlp wp1_resnet2d wp2_3dcnn wp3_convlstm wp4_vit wp5_swin wp6_video_swin)
weather_arch=(weather_nearest weather_bilinear weather_resnet2d weather_3dcnn weather_convlstm weather_vit weather_swin weather_video_swin)
for seed in "${SEEDS[@]}"; do
    for index in "${!weather_ids[@]}"; do
        add_job forecast baselines_weather "${weather_ids[$index]}/seed$seed" "$BASE_CONFIG" NONE \
            "seed=$seed" "model.architecture=${weather_arch[$index]}" \
            model.output_level=direct_cluster \
            training.deterministic.station_loss_weight=0.0 \
            training.deterministic.cluster_loss_weight=2.0 \
            training.deterministic.auxiliary_weather_loss_weight=0.0
    done
done
run_stage weather_baselines "${JOBS[@]}"
select_group "$EXPERIMENT_ROOT/selected/weather_cnn" \
    "$EXPERIMENT_ROOT/baselines_weather/wp1_resnet2d" \
    "$EXPERIMENT_ROOT/baselines_weather/wp2_3dcnn" \
    "$EXPERIMENT_ROOT/baselines_weather/wp3_convlstm"
select_group "$EXPERIMENT_ROOT/selected/weather_transformer" \
    "$EXPERIMENT_ROOT/baselines_weather/wp4_vit" \
    "$EXPERIMENT_ROOT/baselines_weather/wp5_swin" \
    "$EXPERIMENT_ROOT/baselines_weather/wp6_video_swin"

# Hierarchical versions of the best CNN and Transformer weather encoders.
JOBS=()
for seed in "${SEEDS[@]}"; do
    add_job forecast baselines_weather "wp7_best_cnn_hierarchical/seed$seed" \
        "$EXPERIMENT_ROOT/selected/weather_cnn/seed$seed/config.yaml" NONE \
        "seed=$seed" model.output_level=hierarchical \
        training.deterministic.station_loss_weight=1.0 \
        training.deterministic.cluster_loss_weight=2.0 \
        training.deterministic.auxiliary_weather_loss_weight=0.0
    add_job forecast baselines_weather "wp8_best_transformer_hierarchical/seed$seed" \
        "$EXPERIMENT_ROOT/selected/weather_transformer/seed$seed/config.yaml" NONE \
        "seed=$seed" model.output_level=hierarchical \
        training.deterministic.station_loss_weight=1.0 \
        training.deterministic.cluster_loss_weight=2.0 \
        training.deterministic.auxiliary_weather_loss_weight=0.0
done
run_stage weather_hierarchical "${JOBS[@]}"
weather_candidates=()
for id in "${weather_ids[@]}" wp7_best_cnn_hierarchical wp8_best_transformer_hierarchical; do
    weather_candidates+=("$EXPERIMENT_ROOT/baselines_weather/$id")
done
select_group "$EXPERIMENT_ROOT/selected/weather_baseline" "${weather_candidates[@]}"

# MS0-MS6: common encoders and decoder, only the fusion operator changes.
JOBS=()
ms_ids=(ms0_prediction_weight ms1_concat ms2_lead_gate ms3_cross_attention ms4_bidirectional_cross_attention ms5_tft_future_weather ms6_astafn_adaptation)
ms_modes=(prediction_weight direct_concat lead_gate cross_attention bidirectional_cross_attention tft astafn)
for seed in "${SEEDS[@]}"; do
    base="$EXPERIMENT_ROOT/selected/time_alignment/seed$seed/config.yaml"
    for index in "${!ms_ids[@]}"; do
        add_job forecast baselines_fusion "${ms_ids[$index]}/seed$seed" "$base" NONE \
            "seed=$seed" "${common_forced_overrides[@]}" \
            "model.fusion_mode=${ms_modes[$index]}"
    done
done
run_stage fusion_baselines "${JOBS[@]}"
ms_candidates=()
for id in "${ms_ids[@]}"; do ms_candidates+=("$EXPERIMENT_ROOT/baselines_fusion/$id"); done
select_group "$EXPERIMENT_ROOT/selected/fusion_baseline" "${ms_candidates[@]}"

# Freeze the validation-selected final model and expose it under final/seed*.
for seed in "${SEEDS[@]}"; do
    source_dir="$EXPERIMENT_ROOT/selected/training/seed$seed"
    final_dir="$EXPERIMENT_ROOT/final/seed$seed"
    mkdir -p "$final_dir"
    for name in config.yaml best_model.pt metrics_val.json history.json; do
        cp "$source_dir/$name" "$final_dir/$name"
    done
    touch "$final_dir/COMPLETE"
done

# Test is read only after every validation selection is frozen.
JOBS=()
for seed in "${SEEDS[@]}"; do
    add_job test final "seed$seed" NONE NONE 1
    for id in "${history_ids[@]}"; do
        add_job test baselines_history "$id/seed$seed" NONE NONE 0
    done
    for id in "${weather_ids[@]}" wp7_best_cnn_hierarchical wp8_best_transformer_hierarchical; do
        add_job test baselines_weather "$id/seed$seed" NONE NONE 0
    done
    for id in "${ms_ids[@]}"; do
        add_job test baselines_fusion "$id/seed$seed" NONE NONE 0
    done
done
run_stage frozen_test_evaluation "${JOBS[@]}"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/summarize_experiments.py" \
    --root "$EXPERIMENT_ROOT" \
    --output-dir "$EXPERIMENT_ROOT/summary"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/analyze_paper_results.py" \
    --root "$EXPERIMENT_ROOT" \
    --output-dir "$EXPERIMENT_ROOT/analysis"
touch "$EXPERIMENT_ROOT/ALL_COMPLETE"
trap - INT TERM
echo "[paper] all experiments completed root=$EXPERIMENT_ROOT"

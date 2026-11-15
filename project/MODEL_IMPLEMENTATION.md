# Model implementation

## Default data view

The model configuration is `configs/forced_evolution_pad5.yaml`.

- Source archive: `data/jilin_56farm_pad8.h5`
- Model weather crop: station envelope plus 5 native grid points on every side
- Crop bounds: 47.25 to 41.50 degrees north and 120.75 to 126.75 degrees east
- Crop shape: 24 by 25 native 0.25-degree grid points
- Weather input: 8 native hourly nodes, 14 channels
- Power input: 56 stations by 24 observed quarter-hour values
- Power target: 56 stations by 24 future quarter-hour values

The preprocessing does not create spatially interpolated station weather or
quarter-hour weather. It only selects native grid cells and normalizes station
power by each station's nominal capacity.

## Cached splits and loading

Run:

```bash
cd /storage/liying/new_try/wind_cluster_forced_evolution
sh scripts/prepare_model_data.sh
```

The cache is stored in `data/model_cache_pad5_f8`. Small sample arrays and all
power windows are loaded into RAM. Weather cycles are memory-mapped by default,
or preloaded when `data.preload_weather` and its memory budget permit it.

Quarter-hour origins sharing one hourly weather cycle are grouped in a batch.
The pointwise weather encoder, optional regional CNN, and station query process
that weather cycle once, then
`weather_batch_index` maps its representation to the corresponding origins.

## Model path

1. A shared TCN encodes the 24-point power history of every station.
2. A shared pointwise MLP encodes each native grid point independently, without
   mixing neighboring locations.
3. A station-conditioned query aggregates native local grid tokens using station
   coordinates, relative grid displacement, and wind/displacement geometry.
4. An optional second station-conditioned query reads an average-pooled coarse
   grid. A normalized two-way gate exposes the local/regional contribution for
   every station and hourly node.
5. A configurable path encoder processes the eight station forcing nodes.
6. Relative-time attention queries history and weather for each of the 24 target
   timestamps. The weather query uses actual node-to-target time differences.
7. The controlled-transition path initializes a station state from local power,
   cluster power, and station attributes, then updates it for each quarter hour
   using weather control, adjacent control change, target time, and a 0.25-hour step.
8. Station MW predictions are summed exactly to obtain cluster power.

`station_query_mode` separates nearest-point and fixed-bilinear diagnostics from
the learned native-grid query. `station_neighborhood_cells` counts grid cells:
1, 3, and 5 cells use 2 by 2, 4 by 4, and 6 by 6 native grid points,
respectively. `forcing_path_mode`, wind geometry, and the independently encoded
regional-context branch are validation-screened switches.

`regional_pool_size=1` preserves the original local-only path. Values greater
than one enable the coarse-grid branch; `regional_neighborhood_cells` controls
its station-centered support. The model returns native-grid attention,
coarse-grid attention, and two-scale mixture weights separately.

`training.deterministic.ramp_loss_weight` controls optional supervision of all
24 power changes, including the change from the last observation to the first
forecast. A zero value exactly disables the added term.

## Training

Spatial forcing pretraining:

```bash
PYTHONPATH=src /data/liying/anaconda3/envs/chronos/bin/python \
  scripts/train_spatial_pretrain.py --device cuda
```

Full deterministic training initialized from the spatial checkpoint:

```bash
PYTHONPATH=src /data/liying/anaconda3/envs/chronos/bin/python \
  scripts/train_forced_evolution.py \
  --spatial-checkpoint artifacts/spatial_pretrain_pad5/best_model.pt \
  --device cuda
```

The training scripts read only train and validation data. Test evaluation is a
separate command:

```bash
PYTHONPATH=src /data/liying/anaconda3/envs/chronos/bin/python \
  scripts/evaluate_forced_evolution.py \
  --checkpoint artifacts/forced_evolution_pad5/best_model.pt \
  --split test --device cuda
```

The complete parallel shell layout for currently implemented module screening,
training controls, and three-seed evaluation is documented in
`shell/README.md`.

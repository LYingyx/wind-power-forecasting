# Main framework: cross-scale predictive representation and power response

## English caption

**Framework for ultra-short-term wind power forecasting.** (a) Target-farm and regional power observations at 15-minute intervals and hourly weather forecasts provide complementary inputs to a shared forecasting model. The cross-scale representation module constructs predictive features, and the state-conditioned power response module predicts future power changes, which are added to the latest observed power. (b) Conditional low-rank refinement adds a correction to the interpolated affine weather representation. Historical trajectory features, an operating-state summary, window-wide wind changes, and target time are combined through a condition projection, a compact feature representation, and a shared linear projection. (c) The predictive features are projected into a decoder-query component and combined with a weather–time adapter and multi-source context. A separate history encoder supplies decoder memory. Regional readout, operating-state and forecast-feature projections, and site/time projections form the context. All processing networks share weights across farms; a jointly learned identity embedding provides a distinct vector for each farm. Solid arrows denote data flow, and paired dashed lines link the overview modules to their detailed structures.

## Functional names and implementation

| Figure label | Implementation and scope |
|---|---|
| Conditional low-rank refinement | An input-conditioned feature residual; not LoRA or pretrained-weight adaptation |
| Condition projection | `HistoryRefinement.network` and its bounded output activation |
| Low-dimensional features | Compact intermediate features; no separate physical interpretation assigned to their coordinates |
| Shared linear projection | `HistoryRefinement.basis`, a bias-free shared projection into the residual feature space |
| Query projection | `align_out(align_v(Z))`; not a supervised reconstruction of local meteorology |
| Weather–time adapter | The retained neutral adapter takes projected predictive features and time-derived conditions |
| Multi-source context C | Sum of region/neighbor readout, operating/forecast feature projections, and shared site/time projections |
| Farm-specific embedding | One learned identity vector per farm in a shared lookup table, jointly optimized with the network |
| Shared site / time projections | Fixed site attributes and the identity vector enter shared projections; time is a separate conditioning input |

Implementation sources: `project/runs/daily_history_refinement_20261116_v7/code/model.py`, `parent_model.py`, and `backbone.py`.

The state summary **c** used by the representation refiner and decoder context **C** have different contents. Neither is a directly measured local meteorological state. The model has no supervised intermediate local-weather space, no hard capacity clipping, and no subsequent per-farm fine-tuning stage.

## Layout constraints

The decoder, power head and power-change output remain horizontal in the upper-right of panel (c). The lower part contains the actual multi-source context and site-identity pathways, extending into the previously empty lower-right area. The output is not moved downward to fill space. Colors follow the user's v23 reference: pale rose, blue and teal.

The narrowing and widening trapezoids depict condition and linear projections. They do not introduce an encoder–decoder module. Tensor dimensions, layer widths, coefficient labels and matrix-factorization formulas are intentionally omitted from the main paper figure.

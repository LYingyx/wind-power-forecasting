# Response to Reviewers

**Submission ID:** 0084  
**Manuscript title:** Operating-State-Constrained Weather-Power Fusion Network for Regional Wind Farm Cluster Forecasting

Dear BigDIA 2026 Program Committee and Reviewers,

We sincerely thank the Program Committee and the reviewers for accepting our paper subject to revision and for providing careful and constructive comments. We have revised the camera-ready manuscript in response to all comments. In particular, we checked and standardized the typography, reformatted Fig. 5 as a double-column figure, clarified the boundary between the central contribution and established architectural components, simplified the methodological narrative, and strengthened the interpretation of the comparative and ablation experiments. Our point-by-point responses are provided below.

## Response to Reviewer 1

### Comment 1

> The font used in the paper looks somewhat unusual.

### Response

Thank you for pointing this out. We carefully rechecked the typography of the complete manuscript against the IEEE conference format. The revised paper retains the official `IEEEtran` conference class and does not alter the template layout or spacing. Times New Roman is used consistently for the main text, while mathematical symbols use the standard mathematical fonts. The text in all tables has also been standardized to the same `\footnotesize` setting, and the labels and legend in Fig. 5 use Times New Roman at sizes consistent with the surrounding paper. We regenerated the final PDF and verified that all fonts are embedded. These checks remove the inconsistent or unusual appearance noted by the reviewer.

**Location of revision:** Typography throughout the manuscript, including all tables and Fig. 5.

### Comment 2

> Figure 5 could be presented in a double-column format.

### Response

Thank you for this suggestion. Fig. 5 has been reformatted as a full-width double-column figure using the two-column figure environment. Its two panels are now arranged horizontally across the page, which makes the horizon-wise MAE comparison and MAE gap easier to read. The figure has also been placed at the top of page 7 in the revised manuscript.

**Location of revision:** Fig. 5, page 7.

## Response to Reviewer 2

### Comment

> This paper proposes an operating-state-constrained weather-power fusion network for regional wind farm cluster forecasting, integrating weather memory, multi-scale power-state encoding, and a Transformer-based query decoder. The topic is highly relevant to wind power forecasting and spatiotemporal modeling, and the overall framework is well-designed with clear motivation and strong technical depth. The authors conduct experiments on a real-world dataset of 58 wind farms with appropriate baselines and ablation studies, and the results demonstrate consistent improvements over weather-only and CNN/Transformer-based methods. The modeling of forecast-origin power state as an explicit constraint is a meaningful and well-justified idea, and the architecture is generally coherent.
>
> However, the paper is relatively complex and heavily engineered, while the core methodological novelty is moderate, as it mainly combines existing components such as GCNs, cross-attention, and spatiotemporal Transformers. Some design choices are not fully isolated for interpretability, and the contribution could be more clearly simplified. Despite this, the experimental validation is solid and the improvements are consistent across metrics.

### Response

Thank you for this balanced and constructive assessment. We agree that the contribution should not be attributed to the individual GCN, convolutional, cross-attention, or Transformer operators, all of which are established techniques. Our intended contribution is the explicit formulation of recent measured power as a forecast-origin operating-state constraint and the memory-level conditioning mechanism through which this state controls the reading of future-weather information. We have revised the paper to make this boundary explicit, simplify the methodological narrative, and connect each important design choice to a specific experimental control.

The principal revisions are as follows.

1. **The contribution has been refocused on one central mechanism.** In the Introduction (page 1), we now organize the method around three functional stages: weather-context construction, forecast-origin state encoding, and state-conditioned decoding. The contribution list has also been rewritten to distinguish the role of future weather as the main driver of intraday evolution from that of recent measured power as an origin-state constraint.

2. **The method description has been simplified.** In Section IV-A, Overall Framework (page 3), the architecture is explained according to the three information-processing roles above instead of being introduced as a collection of modules. The revised text emphasizes the central information flow: the power-state encoder maps the pre-origin state into dynamic context tokens, and each horizon query reads future weather conditioned on those tokens. The subsequent subsections directly introduce the weather-memory, power-state encoding, and query-decoding structures.

3. **The design choices are now linked to explicit experimental controls.** We reorganized the discussion in Section V-D, Ablation Study (page 7), around four questions: whether the improvement requires true pre-origin power values, whether multi-scale state encoding provides an additional benefit, whether queryable weather memory is necessary, and whether weather remains the dominant driver. Specifically:

   - Weather-only removes the power-state constraint and increases MAE and RMSE by 7.34% and 7.04%, respectively.
   - Random power and Zero power retain the power-side structure but destroy the actual state information. Their degradation is close to that of Weather-only, showing that the improvement comes from valid operating-state content rather than additional parameters or an extra computational path.
   - The output-level fusion baselines in Table I remain approximately 5% worse than the proposed method in both MAE and RMSE. This comparison supports memory-level conditioning rather than merely adding an independent historical-power predictor.
   - Basic GCN increases MAE and RMSE by 4.88% and 4.74%, respectively. We now interpret this as the secondary benefit of multi-scale state encoding, rather than treating the encoder itself as the core novelty.
   - The weather-side ablations, particularly No wind, confirm that future weather remains the primary predictive driver and that the power context acts as a constraint rather than replacing meteorological information.

   These comparisons now form a clear evidence chain: valid origin-state information is necessary, memory-level conditioning is more effective than output-level fusion, multi-scale encoding further strengthens the state representation, and the resulting model remains weather-driven.

4. **The claims and limitations have been made more explicit.** In Section VI, Discussion and Conclusion (page 7), we summarize how information-role assignment and memory-level conditioning coordinate the forecast-origin state with future weather. We also acknowledge that the current evidence is limited to deterministic forecasting in one region and identify cross-region transfer, meteorological uncertainty, model simplification, and extreme-ramp forecasting as future work.

We believe these revisions make the methodological contribution easier to identify and evaluate, while avoiding an overstatement of novelty in the individual architectural components. They also make the existing experimental evidence more interpretable by showing precisely which hypothesis is tested by each baseline or ablation.

We thank the reviewer again for this comment, which helped us substantially improve the focus and clarity of the manuscript.

We hope that the revised manuscript has addressed all concerns and is now suitable for inclusion in the BigDIA 2026 proceedings. We again thank the Program Committee and the reviewers for their time and consideration.

Sincerely,

The Authors

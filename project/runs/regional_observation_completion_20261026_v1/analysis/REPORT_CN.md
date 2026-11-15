# 区域实况补充 D 阶段
完成 2800/2800；2022 拟合、2023 选择，2024 未开。

效应：两 seed 先在站内平均，再风场等权。主指标均取 MAE-selected 权重；负值为改善。

分环拆解在保留 N3 时衡量额外环统计的条件增量；不等于隔离该距离带的全部信息。错配及缺失是敏感性分析，不能单独识别传播因果。

|候选|参考|指标|变化|改善站数|完整风场|
|---|---|---|---:|---:|---:|
|W0/R3_RAW|W0/N0|mae|-0.7749%|47|56|
|W0/R3_RAW|W0/N0|rmse|-0.6688%|43|56|
|W0/R3_RAW|W0/N3|mae|-0.5178%|47|56|
|W0/R3_RAW|W0/N3|rmse|-0.3893%|41|56|
|W0/C2_R3_RAW|W0/N0|mae|+0.0343%|24|56|
|W0/C2_R3_RAW|W0/N0|rmse|-0.0481%|29|56|
|W0/C2_R3_RAW|W0/N3|mae|+0.2935%|11|56|
|W0/C2_R3_RAW|W0/N3|rmse|+0.2332%|21|56|
|W0/C2_R3_RAW|W0/R3_RAW|mae|+0.8155%|8|56|
|W0/C2_R3_RAW|W0/R3_RAW|rmse|+0.6249%|15|56|
|W0/C2_R3_ALLPOOL|W0/N0|mae|+0.0133%|24|56|
|W0/C2_R3_ALLPOOL|W0/N0|rmse|-0.0330%|27|56|
|W0/C2_R3_ALLPOOL|W0/N3|mae|+0.2724%|13|56|
|W0/C2_R3_ALLPOOL|W0/N3|rmse|+0.2483%|19|56|
|W0/C2_R3_ALLPOOL|W0/R3_RAW|mae|+0.7943%|7|56|
|W0/C2_R3_ALLPOOL|W0/R3_RAW|rmse|+0.6401%|14|56|
|W0/SELF_R3_RAW|W0/N0|mae|-0.1076%|36|56|
|W0/SELF_R3_RAW|W0/N0|rmse|-0.1848%|33|56|
|W0/SELF_R3_RAW|W0/N3|mae|+0.1512%|20|56|
|W0/SELF_R3_RAW|W0/N3|rmse|+0.0961%|26|56|
|W0/SELF_R3_RAW|W0/R3_RAW|mae|+0.6725%|9|56|
|W0/SELF_R3_RAW|W0/R3_RAW|rmse|+0.4873%|12|56|
|W0/R3_INNER|W0/N0|mae|-0.2628%|39|56|
|W0/R3_INNER|W0/N0|rmse|-0.2735%|35|56|
|W0/R3_INNER|W0/N3|mae|-0.0044%|25|56|
|W0/R3_INNER|W0/N3|rmse|+0.0072%|29|56|
|W0/R3_INNER|W0/R3_RAW|mae|+0.5161%|14|56|
|W0/R3_INNER|W0/R3_RAW|rmse|+0.3980%|17|56|
|W0/R3_MIDDLE|W0/N0|mae|-0.5185%|43|56|
|W0/R3_MIDDLE|W0/N0|rmse|-0.4823%|42|56|
|W0/R3_MIDDLE|W0/N3|mae|-0.2608%|39|56|
|W0/R3_MIDDLE|W0/N3|rmse|-0.2022%|37|56|
|W0/R3_MIDDLE|W0/R3_RAW|mae|+0.2583%|18|56|
|W0/R3_MIDDLE|W0/R3_RAW|rmse|+0.1878%|19|56|
|W0/R3_OUTER|W0/N0|mae|-0.5886%|40|56|
|W0/R3_OUTER|W0/N0|rmse|-0.5746%|41|56|
|W0/R3_OUTER|W0/N3|mae|-0.3310%|38|56|
|W0/R3_OUTER|W0/N3|rmse|-0.2948%|36|56|
|W0/R3_OUTER|W0/R3_RAW|mae|+0.1877%|15|56|
|W0/R3_OUTER|W0/R3_RAW|rmse|+0.0949%|24|56|
|W0/R3_NO_INNER|W0/N0|mae|-0.7602%|45|56|
|W0/R3_NO_INNER|W0/N0|rmse|-0.6852%|40|56|
|W0/R3_NO_INNER|W0/N3|mae|-0.5031%|45|56|
|W0/R3_NO_INNER|W0/N3|rmse|-0.4057%|38|56|
|W0/R3_NO_INNER|W0/R3_RAW|mae|+0.0148%|19|56|
|W0/R3_NO_INNER|W0/R3_RAW|rmse|-0.0165%|18|56|
|W0/R3_NO_MIDDLE|W0/N0|mae|-0.5764%|43|56|
|W0/R3_NO_MIDDLE|W0/N0|rmse|-0.5438%|41|56|
|W0/R3_NO_MIDDLE|W0/N3|mae|-0.3188%|39|56|
|W0/R3_NO_MIDDLE|W0/N3|rmse|-0.2639%|37|56|
|W0/R3_NO_MIDDLE|W0/R3_RAW|mae|+0.2001%|18|56|
|W0/R3_NO_MIDDLE|W0/R3_RAW|rmse|+0.1259%|19|56|
|W0/R3_NO_OUTER|W0/N0|mae|-0.5319%|42|56|
|W0/R3_NO_OUTER|W0/N0|rmse|-0.5125%|41|56|
|W0/R3_NO_OUTER|W0/N3|mae|-0.2742%|40|56|
|W0/R3_NO_OUTER|W0/N3|rmse|-0.2325%|38|56|
|W0/R3_NO_OUTER|W0/R3_RAW|mae|+0.2448%|17|56|
|W0/R3_NO_OUTER|W0/R3_RAW|rmse|+0.1574%|23|56|
|W0/R3_ALLPOOL|W0/N0|mae|-0.6883%|47|56|
|W0/R3_ALLPOOL|W0/N0|rmse|-0.6654%|44|56|
|W0/R3_ALLPOOL|W0/N3|mae|-0.4310%|47|56|
|W0/R3_ALLPOOL|W0/N3|rmse|-0.3858%|37|56|
|W0/R3_ALLPOOL|W0/R3_RAW|mae|+0.0872%|19|56|
|W0/R3_ALLPOOL|W0/R3_RAW|rmse|+0.0035%|24|56|
|W0/R3_ALLPOOL_MEAN|W0/N0|mae|-0.4818%|42|56|
|W0/R3_ALLPOOL_MEAN|W0/N0|rmse|-0.4741%|39|56|
|W0/R3_ALLPOOL_MEAN|W0/N3|mae|-0.2240%|35|56|
|W0/R3_ALLPOOL_MEAN|W0/N3|rmse|-0.1940%|35|56|
|W0/R3_ALLPOOL_MEAN|W0/R3_RAW|mae|+0.2954%|15|56|
|W0/R3_ALLPOOL_MEAN|W0/R3_RAW|rmse|+0.1960%|21|56|
|W0/R3_NO_N3|W0/N0|mae|-0.7905%|49|56|
|W0/R3_NO_N3|W0/N0|rmse|-0.6839%|45|56|
|W0/R3_NO_N3|W0/N3|mae|-0.5335%|48|56|
|W0/R3_NO_N3|W0/N3|rmse|-0.4044%|40|56|
|W0/R3_NO_N3|W0/R3_RAW|mae|-0.0158%|37|56|
|W0/R3_NO_N3|W0/R3_RAW|rmse|-0.0152%|33|56|
|W0/R3_MEAN|W0/N0|mae|-0.5833%|44|56|
|W0/R3_MEAN|W0/N0|rmse|-0.6005%|44|56|
|W0/R3_MEAN|W0/N3|mae|-0.3257%|42|56|
|W0/R3_MEAN|W0/N3|rmse|-0.3208%|41|56|
|W0/R3_MEAN|W0/R3_RAW|mae|+0.1930%|19|56|
|W0/R3_MEAN|W0/R3_RAW|rmse|+0.0687%|24|56|
|W0/R3_MEAN_DELTA|W0/N0|mae|-0.8649%|48|56|
|W0/R3_MEAN_DELTA|W0/N0|rmse|-0.7815%|48|56|
|W0/R3_MEAN_DELTA|W0/N3|mae|-0.6080%|49|56|
|W0/R3_MEAN_DELTA|W0/N3|rmse|-0.5022%|43|56|
|W0/R3_MEAN_DELTA|W0/R3_RAW|mae|-0.0907%|38|56|
|W0/R3_MEAN_DELTA|W0/R3_RAW|rmse|-0.1134%|37|56|
|W0/R3_D0|W0/N0|mae|-1.2012%|52|56|
|W0/R3_D0|W0/N0|rmse|-1.0547%|50|56|
|W0/R3_D0|W0/N3|mae|-0.9453%|50|56|
|W0/R3_D0|W0/N3|rmse|-0.7762%|50|56|
|W0/R3_D0|W0/R3_RAW|mae|-0.4297%|48|56|
|W0/R3_D0|W0/R3_RAW|rmse|-0.3884%|47|56|
|W0/R3_D30|W0/N0|mae|-0.5634%|43|56|
|W0/R3_D30|W0/N0|rmse|-0.4750%|38|56|
|W0/R3_D30|W0/N3|mae|-0.3058%|40|56|
|W0/R3_D30|W0/N3|rmse|-0.1949%|33|56|
|W0/R3_D30|W0/R3_RAW|mae|+0.2131%|7|56|
|W0/R3_D30|W0/R3_RAW|rmse|+0.1952%|15|56|
|W0/R3_RAW|W0/C2_R3_RAW|mae|-0.8089%|48|56|
|W0/R3_RAW|W0/C2_R3_RAW|rmse|-0.6210%|41|56|
|W0/R3_ALLPOOL|W0/C2_R3_ALLPOOL|mae|-0.7016%|49|56|
|W0/R3_ALLPOOL|W0/C2_R3_ALLPOOL|rmse|-0.6326%|43|56|
|W0/R3_RAW|W0/SELF_R3_RAW|mae|-0.6680%|47|56|
|W0/R3_RAW|W0/SELF_R3_RAW|rmse|-0.4849%|44|56|
|W0/R3_ALLPOOL_MEAN|W0/R3_MEAN|mae|+0.1021%|14|56|
|W0/R3_ALLPOOL_MEAN|W0/R3_MEAN|rmse|+0.1272%|17|56|
|TF_z1_lr0.0002_dp0.05/R3_RAW|TF_z1_lr0.0002_dp0.05/N0|mae|-0.9224%|52|56|
|TF_z1_lr0.0002_dp0.05/R3_RAW|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.7664%|47|56|
|TF_z1_lr0.0002_dp0.05/R3_RAW|TF_z1_lr0.0002_dp0.05/N3|mae|-0.7130%|47|56|
|TF_z1_lr0.0002_dp0.05/R3_RAW|TF_z1_lr0.0002_dp0.05/N3|rmse|-0.5617%|43|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_RAW|TF_z1_lr0.0002_dp0.05/N0|mae|+0.0268%|25|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_RAW|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.0350%|30|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_RAW|TF_z1_lr0.0002_dp0.05/N3|mae|+0.2382%|17|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_RAW|TF_z1_lr0.0002_dp0.05/N3|rmse|+0.1711%|23|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_RAW|TF_z1_lr0.0002_dp0.05/R3_RAW|mae|+0.9581%|2|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_RAW|TF_z1_lr0.0002_dp0.05/R3_RAW|rmse|+0.7370%|12|56|
|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/N0|mae|-0.8245%|54|56|
|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.7163%|51|56|
|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/N3|mae|-0.6149%|50|56|
|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/N3|rmse|-0.5116%|47|56|
|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/R3_RAW|mae|+0.0989%|23|56|
|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/R3_RAW|rmse|+0.0505%|29|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/N0|mae|+0.0041%|28|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.0442%|28|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/N3|mae|+0.2154%|18|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/N3|rmse|+0.1619%|22|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/R3_RAW|mae|+0.9351%|2|56|
|TF_z1_lr0.0002_dp0.05/C2_R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/R3_RAW|rmse|+0.7278%|10|56|
|TF_z1_lr0.0002_dp0.05/R3_RAW|TF_z1_lr0.0002_dp0.05/C2_R3_RAW|mae|-0.9490%|54|56|
|TF_z1_lr0.0002_dp0.05/R3_RAW|TF_z1_lr0.0002_dp0.05/C2_R3_RAW|rmse|-0.7316%|44|56|
|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/C2_R3_ALLPOOL|mae|-0.8285%|52|56|
|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/C2_R3_ALLPOOL|rmse|-0.6724%|50|56|
|W0/N0|TF_z1_lr0.0002_dp0.05/N0|mae|-1.4843%|51|56|
|W0/N0|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.7867%|39|56|
|W0/N3|TF_z1_lr0.0002_dp0.05/N3|mae|-1.5312%|51|56|
|W0/N3|TF_z1_lr0.0002_dp0.05/N3|rmse|-0.8611%|42|56|
|W0/R3_RAW|TF_z1_lr0.0002_dp0.05/R3_RAW|mae|-1.3376%|50|56|
|W0/R3_RAW|TF_z1_lr0.0002_dp0.05/R3_RAW|rmse|-0.6892%|38|56|
|W0/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|mae|-1.3491%|51|56|
|W0/R3_ALLPOOL|TF_z1_lr0.0002_dp0.05/R3_ALLPOOL|rmse|-0.7358%|40|56|

预设判定：
```json
{
  "timestamp": "2026-10-27T02:16:09.095135+08:00",
  "complete": true,
  "primary": "R3_RAW",
  "gates": {
    "W0/R3_RAW": {
      "comparisons": {
        "N0": {
          "mae": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08471579682713355,
            "reference": 0.08537736473214172,
            "effect_percent": -0.7748750586103692,
            "improved_farms": 47,
            "improved_seed42": 45,
            "improved_seed43": 46
          },
          "rmse": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.13219658974596166,
            "reference": 0.13308671922596207,
            "effect_percent": -0.6688341896001759,
            "improved_farms": 43,
            "improved_seed42": 43,
            "improved_seed43": 42
          }
        },
        "N3": {
          "mae": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08471579682713355,
            "reference": 0.08515673410525901,
            "effect_percent": -0.5177949609721177,
            "improved_farms": 47,
            "improved_seed42": 44,
            "improved_seed43": 45
          },
          "rmse": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.13219658974596166,
            "reference": 0.13271323822958958,
            "effect_percent": -0.389296870847311,
            "improved_farms": 41,
            "improved_seed42": 33,
            "improved_seed43": 39
          }
        },
        "C2_R3_RAW": {
          "mae": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08471579682713355,
            "reference": 0.08540667261142491,
            "effect_percent": -0.808924833583724,
            "improved_farms": 48,
            "improved_seed42": 47,
            "improved_seed43": 46
          },
          "rmse": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.13219658974596166,
            "reference": 0.13302269199197267,
            "effect_percent": -0.6210235514259921,
            "improved_farms": 41,
            "improved_seed42": 39,
            "improved_seed43": 40
          }
        }
      },
      "dual": true,
      "beats_control": true,
      "high_power_rmse_percent": -0.010720243163564458,
      "passes": true
    },
    "W0/R3_ALLPOOL": {
      "comparisons": {
        "N0": {
          "mae": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08478967009899484,
            "reference": 0.08537736473214172,
            "effect_percent": -0.6883494647447641,
            "improved_farms": 47,
            "improved_seed42": 45,
            "improved_seed43": 47
          },
          "rmse": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.1322012131089337,
            "reference": 0.13308671922596207,
            "effect_percent": -0.6653602419373739,
            "improved_farms": 44,
            "improved_seed42": 43,
            "improved_seed43": 44
          }
        },
        "N3": {
          "mae": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08478967009899484,
            "reference": 0.08515673410525901,
            "effect_percent": -0.4310451899323109,
            "improved_farms": 47,
            "improved_seed42": 46,
            "improved_seed43": 40
          },
          "rmse": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.1322012131089337,
            "reference": 0.13271323822958958,
            "effect_percent": -0.38581314681666035,
            "improved_farms": 37,
            "improved_seed42": 39,
            "improved_seed43": 40
          }
        },
        "C2_R3_ALLPOOL": {
          "mae": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08478967009899484,
            "reference": 0.08538872046816226,
            "effect_percent": -0.701556793312974,
            "improved_farms": 49,
            "improved_seed42": 47,
            "improved_seed43": 45
          },
          "rmse": {
            "candidate_configuration": "W0",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "W0",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.1322012131089337,
            "reference": 0.1330428307403237,
            "effect_percent": -0.6325914945636391,
            "improved_farms": 43,
            "improved_seed42": 43,
            "improved_seed43": 40
          }
        }
      },
      "dual": true,
      "beats_control": true,
      "high_power_rmse_percent": -0.17431966990018033,
      "passes": true
    },
    "TF_z1_lr0.0002_dp0.05/R3_RAW": {
      "comparisons": {
        "N0": {
          "mae": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08586432503207732,
            "reference": 0.08666373604522692,
            "effect_percent": -0.9224285146585687,
            "improved_farms": 52,
            "improved_seed42": 48,
            "improved_seed43": 53
          },
          "rmse": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.1331140494558942,
            "reference": 0.13414207547561613,
            "effect_percent": -0.76637104061269,
            "improved_farms": 47,
            "improved_seed42": 43,
            "improved_seed43": 51
          }
        },
        "N3": {
          "mae": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08586432503207732,
            "reference": 0.08648095985971582,
            "effect_percent": -0.7130295832039502,
            "improved_farms": 47,
            "improved_seed42": 46,
            "improved_seed43": 51
          },
          "rmse": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.1331140494558942,
            "reference": 0.1338660095513708,
            "effect_percent": -0.5617259362527216,
            "improved_farms": 43,
            "improved_seed42": 41,
            "improved_seed43": 45
          }
        },
        "C2_R3_RAW": {
          "mae": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08586432503207732,
            "reference": 0.08668697476272073,
            "effect_percent": -0.948988856624855,
            "improved_farms": 54,
            "improved_seed42": 50,
            "improved_seed43": 53
          },
          "rmse": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_RAW",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.1331140494558942,
            "reference": 0.1340951114673584,
            "effect_percent": -0.7316165374924966,
            "improved_farms": 44,
            "improved_seed42": 40,
            "improved_seed43": 47
          }
        }
      },
      "dual": true,
      "beats_control": true,
      "high_power_rmse_percent": -0.04717909308806778,
      "passes": true
    },
    "TF_z1_lr0.0002_dp0.05/R3_ALLPOOL": {
      "comparisons": {
        "N0": {
          "mae": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08594922057128447,
            "reference": 0.08666373604522692,
            "effect_percent": -0.8244688107717502,
            "improved_farms": 54,
            "improved_seed42": 50,
            "improved_seed43": 55
          },
          "rmse": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.13318121346418552,
            "reference": 0.13414207547561613,
            "effect_percent": -0.7163017330869326,
            "improved_farms": 51,
            "improved_seed42": 46,
            "improved_seed43": 51
          }
        },
        "N3": {
          "mae": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08594922057128447,
            "reference": 0.08648095985971582,
            "effect_percent": -0.6148628429817449,
            "improved_farms": 50,
            "improved_seed42": 45,
            "improved_seed43": 52
          },
          "rmse": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.13318121346418552,
            "reference": 0.1338660095513708,
            "effect_percent": -0.5115533730184807,
            "improved_farms": 47,
            "improved_seed42": 42,
            "improved_seed43": 47
          }
        },
        "C2_R3_ALLPOOL": {
          "mae": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.08594922057128447,
            "reference": 0.08666726898525108,
            "effect_percent": -0.828511642715779,
            "improved_farms": 52,
            "improved_seed42": 50,
            "improved_seed43": 54
          },
          "rmse": {
            "candidate_configuration": "TF_z1_lr0.0002_dp0.05",
            "candidate_variant": "R3_ALLPOOL",
            "reference_configuration": "TF_z1_lr0.0002_dp0.05",
            "selection": "mae",
            "horizon": "all",
            "farms": 56,
            "pairs": 112,
            "candidate": 0.13318121346418552,
            "reference": 0.13408280083933607,
            "effect_percent": -0.6724109054306537,
            "improved_farms": 50,
            "improved_seed42": 47,
            "improved_seed43": 49
          }
        }
      },
      "dual": true,
      "beats_control": true,
      "high_power_rmse_percent": -0.4095578757695062,
      "passes": true
    }
  },
  "simplification": {
    "mae": {
      "upper_one_sided_95_percent": 0.17806236171516107,
      "passes": false
    },
    "rmse": {
      "upper_one_sided_95_percent": 0.10760813656293443,
      "passes": false
    }
  },
  "recommended_variant": "R3_RAW",
  "open_2024": false,
  "next": "Review D controls and final E input-factor definitions; freeze E comparison manifest before evaluating 2024.",
  "interpretation": "2023 selection evidence, not causal proof; fixed-weight perturbations can be out of distribution"
}
```

# 区域残差筛选
完成 2968/2968；2022 拟合、2023 选择，未开 2024。

效应量：站内两 seed 平均，再对风场等权，报告候选均值/参考均值−1。所有原值保留；小于 0.1% 的变化也报告。分组只使用两 seed 齐全且可配对的风场。

残差包含功率映射和运行因素，相关性与固定权重干预不能单独证明气象误差传播；筛选年份上的优势仍有选择偏差。

|候选|参考|指标|变化|改善风场|完整风场|
|---|---|---|---:|---:|---:|
|W0/R1|W0/N0|mae|-0.3964%|44|56|
|W0/R1|W0/N0|rmse|-0.3682%|39|56|
|W0/R1|W0/N1|mae|-0.1172%|36|56|
|W0/R1|W0/N1|rmse|-0.0869%|32|56|
|W0/R1|W0/N3|mae|-0.1383%|35|56|
|W0/R1|W0/N3|rmse|-0.0879%|32|56|
|W0/R1|W0/C1_R1|mae|-0.3737%|44|56|
|W0/R1|W0/C1_R1|rmse|-0.3366%|39|56|
|W0/R1|W0/C2_R1|mae|-0.4156%|44|56|
|W0/R1|W0/C2_R1|rmse|-0.3632%|39|56|
|W0/R2|W0/N0|mae|-0.1601%|34|56|
|W0/R2|W0/N0|rmse|-0.1774%|33|56|
|W0/R2|W0/N1|mae|+0.1198%|20|56|
|W0/R2|W0/N1|rmse|+0.1045%|20|56|
|W0/R2|W0/N3|mae|+0.0986%|20|56|
|W0/R2|W0/N3|rmse|+0.1035%|21|56|
|W0/R2|W0/C1_R2|mae|-0.1049%|31|56|
|W0/R2|W0/C1_R2|rmse|-0.1282%|30|56|
|W0/R2|W0/C2_R2|mae|-0.1300%|34|56|
|W0/R2|W0/C2_R2|rmse|-0.1193%|30|56|
|W0/R3|W0/N0|mae|-0.6072%|39|56|
|W0/R3|W0/N0|rmse|-0.4861%|37|56|
|W0/R3|W0/N1|mae|-0.3287%|35|56|
|W0/R3|W0/N1|rmse|-0.2051%|34|56|
|W0/R3|W0/N3|mae|-0.3497%|36|56|
|W0/R3|W0/N3|rmse|-0.2060%|32|56|
|W0/R3|W0/C1_R3|mae|-0.6270%|44|56|
|W0/R3|W0/C1_R3|rmse|-0.5396%|45|56|
|W0/R3|W0/C2_R3|mae|-0.7098%|42|56|
|W0/R3|W0/C2_R3|rmse|-0.5175%|41|56|
|W0/R4|W0/N0|mae|-0.1974%|31|56|
|W0/R4|W0/N0|rmse|-0.1057%|26|56|
|W0/R4|W0/N1|mae|+0.0823%|24|56|
|W0/R4|W0/N1|rmse|+0.1764%|24|56|
|W0/R4|W0/N3|mae|+0.0612%|25|56|
|W0/R4|W0/N3|rmse|+0.1754%|25|56|
|W0/R4|W0/C1_R4|mae|-0.2172%|32|56|
|W0/R4|W0/C1_R4|rmse|-0.1595%|32|56|
|W0/R4|W0/C2_R4|mae|-0.3287%|37|56|
|W0/R4|W0/C2_R4|rmse|-0.1487%|29|56|
|W0/R1|W0/R1_RAWMAP|mae|-0.0209%|29|56|
|W0/R1|W0/R1_RAWMAP|rmse|-0.1018%|28|56|
|W0/R1|W0/R1_SELF|mae|-0.2491%|38|56|
|W0/R1|W0/R1_SELF|rmse|-0.1751%|33|56|
|W0/R2|W0/R2_PAIR|mae|-0.0171%|29|56|
|W0/R2|W0/R2_PAIR|rmse|+0.0111%|30|56|
|W0/R3|W0/R3_RAW|mae|+0.1689%|18|56|
|W0/R3|W0/R3_RAW|rmse|+0.1840%|19|56|
|W0/R3|W0/R1|mae|-0.2117%|35|56|
|W0/R3|W0/R1|rmse|-0.1182%|34|56|
|W0/R4|W0/R3|mae|+0.4124%|6|56|
|W0/R4|W0/R3|rmse|+0.3822%|11|56|
|TF_z1_lr0.0002_dp0.05/R1|TF_z1_lr0.0002_dp0.05/N0|mae|-0.4432%|44|56|
|TF_z1_lr0.0002_dp0.05/R1|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.4182%|46|56|
|TF_z1_lr0.0002_dp0.05/R1|TF_z1_lr0.0002_dp0.05/N1|mae|-0.1994%|40|56|
|TF_z1_lr0.0002_dp0.05/R1|TF_z1_lr0.0002_dp0.05/N1|rmse|-0.1552%|39|56|
|TF_z1_lr0.0002_dp0.05/R1|TF_z1_lr0.0002_dp0.05/N3|mae|-0.2328%|38|56|
|TF_z1_lr0.0002_dp0.05/R1|TF_z1_lr0.0002_dp0.05/N3|rmse|-0.2127%|38|56|
|TF_z1_lr0.0002_dp0.05/R2|TF_z1_lr0.0002_dp0.05/N0|mae|-0.1487%|32|56|
|TF_z1_lr0.0002_dp0.05/R2|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.2206%|42|56|
|TF_z1_lr0.0002_dp0.05/R2|TF_z1_lr0.0002_dp0.05/N1|mae|+0.0958%|19|56|
|TF_z1_lr0.0002_dp0.05/R2|TF_z1_lr0.0002_dp0.05/N1|rmse|+0.0430%|30|56|
|TF_z1_lr0.0002_dp0.05/R2|TF_z1_lr0.0002_dp0.05/N3|mae|+0.0623%|20|56|
|TF_z1_lr0.0002_dp0.05/R2|TF_z1_lr0.0002_dp0.05/N3|rmse|-0.0146%|27|56|
|TF_z1_lr0.0002_dp0.05/R3|TF_z1_lr0.0002_dp0.05/N0|mae|-0.8427%|49|56|
|TF_z1_lr0.0002_dp0.05/R3|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.7378%|40|56|
|TF_z1_lr0.0002_dp0.05/R3|TF_z1_lr0.0002_dp0.05/N1|mae|-0.5999%|45|56|
|TF_z1_lr0.0002_dp0.05/R3|TF_z1_lr0.0002_dp0.05/N1|rmse|-0.4756%|40|56|
|TF_z1_lr0.0002_dp0.05/R3|TF_z1_lr0.0002_dp0.05/N3|mae|-0.6332%|43|56|
|TF_z1_lr0.0002_dp0.05/R3|TF_z1_lr0.0002_dp0.05/N3|rmse|-0.5329%|38|56|
|TF_z1_lr0.0002_dp0.05/R4|TF_z1_lr0.0002_dp0.05/N0|mae|-0.5493%|38|56|
|TF_z1_lr0.0002_dp0.05/R4|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.4689%|37|56|
|TF_z1_lr0.0002_dp0.05/R4|TF_z1_lr0.0002_dp0.05/N1|mae|-0.3058%|34|56|
|TF_z1_lr0.0002_dp0.05/R4|TF_z1_lr0.0002_dp0.05/N1|rmse|-0.2060%|33|56|
|TF_z1_lr0.0002_dp0.05/R4|TF_z1_lr0.0002_dp0.05/N3|mae|-0.3391%|35|56|
|TF_z1_lr0.0002_dp0.05/R4|TF_z1_lr0.0002_dp0.05/N3|rmse|-0.2635%|30|56|
|W0/N0|TF_z1_lr0.0002_dp0.05/N0|mae|-1.4843%|51|56|
|W0/N0|TF_z1_lr0.0002_dp0.05/N0|rmse|-0.7869%|39|56|
|W0/N1|TF_z1_lr0.0002_dp0.05/N1|mae|-1.5191%|51|56|
|W0/N1|TF_z1_lr0.0002_dp0.05/N1|rmse|-0.8049%|40|56|
|W0/N3|TF_z1_lr0.0002_dp0.05/N3|mae|-1.5312%|51|56|
|W0/N3|TF_z1_lr0.0002_dp0.05/N3|rmse|-0.8611%|42|56|
|W0/R1|TF_z1_lr0.0002_dp0.05/R1|mae|-1.4380%|51|56|
|W0/R1|TF_z1_lr0.0002_dp0.05/R1|rmse|-0.7371%|40|56|
|W0/R2|TF_z1_lr0.0002_dp0.05/R2|mae|-1.4955%|51|56|
|W0/R2|TF_z1_lr0.0002_dp0.05/R2|rmse|-0.7440%|41|56|
|W0/R3|TF_z1_lr0.0002_dp0.05/R3|mae|-1.2504%|49|56|
|W0/R3|TF_z1_lr0.0002_dp0.05/R3|rmse|-0.5353%|37|56|
|W0/R4|TF_z1_lr0.0002_dp0.05/R4|mae|-1.1357%|48|56|
|W0/R4|TF_z1_lr0.0002_dp0.05/R4|rmse|-0.4248%|36|56|

门槛判断：
{
  "R1": {
    "comparisons": {
      "N1": {
        "mae": {
          "effect_percent": -0.11723807428486444,
          "improved_farms": 36
        },
        "rmse": {
          "effect_percent": -0.08692624402051319,
          "improved_farms": 32
        }
      },
      "N3": {
        "mae": {
          "effect_percent": -0.1383252018480552,
          "improved_farms": 35
        },
        "rmse": {
          "effect_percent": -0.08786251887648033,
          "improved_farms": 32
        }
      },
      "C1_R1": {
        "mae": {
          "effect_percent": -0.3737312407829041,
          "improved_farms": 44
        },
        "rmse": {
          "effect_percent": -0.33659835291383056,
          "improved_farms": 39
        }
      },
      "C2_R1": {
        "mae": {
          "effect_percent": -0.4156054326999614,
          "improved_farms": 44
        },
        "rmse": {
          "effect_percent": -0.3631575541824361,
          "improved_farms": 39
        }
      }
    },
    "dual_metric_gate": false,
    "beats_matched_controls": true,
    "high_power_rmse_vs_N3_percent": 0.16714584306052327,
    "high_power_gate": false,
    "passes": false
  },
  "R2": {
    "comparisons": {
      "N1": {
        "mae": {
          "effect_percent": 0.11975897904059618,
          "improved_farms": 20
        },
        "rmse": {
          "effect_percent": 0.10445796995364187,
          "improved_farms": 20
        }
      },
      "N3": {
        "mae": {
          "effect_percent": 0.09862181694693106,
          "improved_farms": 20
        },
        "rmse": {
          "effect_percent": 0.10351990165642366,
          "improved_farms": 21
        }
      },
      "C1_R2": {
        "mae": {
          "effect_percent": -0.10493350386087119,
          "improved_farms": 31
        },
        "rmse": {
          "effect_percent": -0.12823981479539448,
          "improved_farms": 30
        }
      },
      "C2_R2": {
        "mae": {
          "effect_percent": -0.1299922410556542,
          "improved_farms": 34
        },
        "rmse": {
          "effect_percent": -0.11929052859860345,
          "improved_farms": 30
        }
      }
    },
    "dual_metric_gate": false,
    "beats_matched_controls": true,
    "high_power_rmse_vs_N3_percent": 0.11832296582745538,
    "high_power_gate": false,
    "passes": false
  },
  "R3": {
    "comparisons": {
      "N1": {
        "mae": {
          "effect_percent": -0.32868494854215724,
          "improved_farms": 35
        },
        "rmse": {
          "effect_percent": -0.20506572199791018,
          "improved_farms": 34
        }
      },
      "N3": {
        "mae": {
          "effect_percent": -0.3497274356976776,
          "improved_farms": 36
        },
        "rmse": {
          "effect_percent": -0.20600088978131037,
          "improved_farms": 32
        }
      },
      "C1_R3": {
        "mae": {
          "effect_percent": -0.6269541255387701,
          "improved_farms": 44
        },
        "rmse": {
          "effect_percent": -0.5396317752197621,
          "improved_farms": 45
        }
      },
      "C2_R3": {
        "mae": {
          "effect_percent": -0.7098457530280489,
          "improved_farms": 42
        },
        "rmse": {
          "effect_percent": -0.5174831215414843,
          "improved_farms": 41
        }
      }
    },
    "dual_metric_gate": true,
    "beats_matched_controls": true,
    "high_power_rmse_vs_N3_percent": 0.004790809945065533,
    "high_power_gate": true,
    "passes": true
  },
  "R4": {
    "comparisons": {
      "N1": {
        "mae": {
          "effect_percent": 0.08233549060085998,
          "improved_farms": 24
        },
        "rmse": {
          "effect_percent": 0.17637895517810254,
          "improved_farms": 24
        }
      },
      "N3": {
        "mae": {
          "effect_percent": 0.06120622930869679,
          "improved_farms": 25
        },
        "rmse": {
          "effect_percent": 0.175440212916933,
          "improved_farms": 25
        }
      },
      "C1_R4": {
        "mae": {
          "effect_percent": -0.21716367646911205,
          "improved_farms": 32
        },
        "rmse": {
          "effect_percent": -0.15946590483979506,
          "improved_farms": 32
        }
      },
      "C2_R4": {
        "mae": {
          "effect_percent": -0.3286839850554002,
          "improved_farms": 37
        },
        "rmse": {
          "effect_percent": -0.14865666912475817,
          "improved_farms": 29
        }
      }
    },
    "dual_metric_gate": false,
    "beats_matched_controls": true,
    "high_power_rmse_vs_N3_percent": 0.4370600093156263,
    "high_power_gate": false,
    "passes": false
  }
}

ARX 不复制随机 seed；LOCAL/REG_RAW/REG_RES/REG_PAIR 使用相同标准化、稀疏算法与选择网格。

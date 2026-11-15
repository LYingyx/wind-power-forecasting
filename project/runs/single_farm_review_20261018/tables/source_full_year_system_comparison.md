| model | split | context | training | mae | rmse_macro | r2_macro | seed_sd | farm_sd | farms | seeds | records | precision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CBH_IC | test | regional history | independent | 0.087228 |  |  | 0.000116 | 0.018899 | 56 | 10 | 560 | archived BF16 / see protocol |
| CBH_IC | val | regional history | independent | 0.088445 |  |  | 0.000101 | 0.015581 | 56 | 10 | 560 | archived BF16 / see protocol |
| CBH_IL | test | local history / weather only | independent | 0.086998 |  |  | 0.000111 | 0.019283 | 56 | 10 | 560 | archived BF16 / see protocol |
| CBH_IL | val | local history / weather only | independent | 0.088450 |  |  | 0.000129 | 0.015996 | 56 | 10 | 560 | archived BF16 / see protocol |
| CBH_SC | test | regional history | shared across source56 | 0.081507 |  |  | 0.000318 | 0.017219 | 56 | 10 | 560 | archived BF16 / see protocol |
| CBH_SC | val | regional history | shared across source56 | 0.083814 |  |  | 0.000303 | 0.015109 | 56 | 10 | 560 | archived BF16 / see protocol |
| CBH_SL | test | local history / weather only | shared across source56 | 0.081799 |  |  | 0.000261 | 0.017396 | 56 | 10 | 560 | archived BF16 / see protocol |
| CBH_SL | val | local history / weather only | shared across source56 | 0.084140 |  |  | 0.000179 | 0.015325 | 56 | 10 | 560 | archived BF16 / see protocol |
| CNN_W | test | local history / weather only | shared across source56 | 0.115232 | 0.170227 | 0.404022 | 0.000379 | 0.028145 | 56 | 10 | 560 | archived BF16 / see protocol |
| CNN_W | val | local history / weather only | shared across source56 | 0.122563 | 0.176882 | 0.443692 | 0.000627 | 0.032250 | 56 | 10 | 560 | archived BF16 / see protocol |
| DLINEAR_FC | test | local history / weather only | shared across source56 | 0.108997 | 0.162071 | 0.501956 | 0.000187 | 0.021556 | 56 | 10 | 560 | archived BF16 / see protocol |
| DLINEAR_FC | val | local history / weather only | shared across source56 | 0.112123 | 0.166440 | 0.516924 | 0.000140 | 0.020922 | 56 | 10 | 560 | archived BF16 / see protocol |
| E0 | test | regional history | shared across source56 | 0.081528 | 0.128062 | 0.685668 | 0.000242 | 0.017323 | 56 | 10 | 560 | FP32 |
| E0 | val | regional history | shared across source56 | 0.083827 | 0.131481 | 0.692973 | 0.000220 | 0.015179 | 56 | 10 | 560 | FP32 |
| E2C | test | regional history | shared across source56 | 0.080703 | 0.127069 | 0.689828 | 0.000126 | 0.017317 | 56 | 10 | 560 | FP32 |
| E2C | val | regional history | shared across source56 | 0.082667 | 0.129898 | 0.699847 | 0.000129 | 0.014904 | 56 | 10 | 560 | FP32 |
| GRU_H | test | local history / weather only | shared across source56 | 0.106572 | 0.163146 | 0.498368 | 0.000078 | 0.022569 | 56 | 10 | 560 | archived BF16 / see protocol |
| GRU_H | val | local history / weather only | shared across source56 | 0.111366 | 0.169090 | 0.505278 | 0.000035 | 0.022424 | 56 | 10 | 560 | archived BF16 / see protocol |
| ITRANSFORMER_H | test | local history / weather only | shared across source56 | 0.106771 | 0.163770 | 0.494420 | 0.000034 | 0.022714 | 56 | 10 | 560 | archived BF16 / see protocol |
| ITRANSFORMER_H | val | local history / weather only | shared across source56 | 0.111667 | 0.169860 | 0.500715 | 0.000032 | 0.022609 | 56 | 10 | 560 | archived BF16 / see protocol |
| LSTM_H | test | local history / weather only | shared across source56 | 0.106594 | 0.163259 | 0.497759 | 0.000104 | 0.022661 | 56 | 10 | 560 | archived BF16 / see protocol |
| LSTM_H | val | local history / weather only | shared across source56 | 0.111398 | 0.169174 | 0.504921 | 0.000058 | 0.022554 | 56 | 10 | 560 | archived BF16 / see protocol |
| MODERNTCN_H | test | local history / weather only | shared across source56 | 0.106458 | 0.163121 | 0.498542 | 0.000085 | 0.022581 | 56 | 10 | 560 | archived BF16 / see protocol |
| MODERNTCN_H | val | local history / weather only | shared across source56 | 0.111295 | 0.169113 | 0.505156 | 0.000062 | 0.022433 | 56 | 10 | 560 | archived BF16 / see protocol |
| NBEATSX_FC | test | local history / weather only | shared across source56 | 0.085961 | 0.132098 | 0.666397 | 0.000360 | 0.017633 | 56 | 10 | 560 | archived BF16 / see protocol |
| NBEATSX_FC | val | local history / weather only | shared across source56 | 0.089040 | 0.136094 | 0.673081 | 0.000248 | 0.016491 | 56 | 10 | 560 | archived BF16 / see protocol |
| PATCHTST_H | test | local history / weather only | shared across source56 | 0.106675 | 0.163309 | 0.497255 | 0.000054 | 0.022591 | 56 | 10 | 560 | archived BF16 / see protocol |
| PATCHTST_H | val | local history / weather only | shared across source56 | 0.111493 | 0.169280 | 0.504068 | 0.000025 | 0.022453 | 56 | 10 | 560 | archived BF16 / see protocol |
| Persistence | test | local history / weather only | none | 0.112763 | 0.175619 | 0.415666 |  | 0.023363 | 56 | 1 | 56 | deterministic |
| Persistence | val | local history / weather only | none | 0.117542 | 0.182200 | 0.421143 |  | 0.022348 | 56 | 1 | 56 | deterministic |
| SWIN_W | test | local history / weather only | shared across source56 | 0.114129 | 0.168659 | 0.412168 | 0.000749 | 0.028155 | 56 | 10 | 560 | archived BF16 / see protocol |
| SWIN_W | val | local history / weather only | shared across source56 | 0.121518 | 0.175277 | 0.454379 | 0.000529 | 0.032394 | 56 | 10 | 560 | archived BF16 / see protocol |
| TCN_FC | test | local history / weather only | shared across source56 | 0.086435 | 0.135134 | 0.650038 | 0.000316 | 0.018317 | 56 | 10 | 560 | archived BF16 / see protocol |
| TCN_FC | val | local history / weather only | shared across source56 | 0.090184 | 0.140007 | 0.655498 | 0.000228 | 0.017553 | 56 | 10 | 560 | archived BF16 / see protocol |
| TFT_FC | test | local history / weather only | shared across source56 | 0.085779 | 0.134527 | 0.653636 | 0.000561 | 0.017791 | 56 | 10 | 560 | archived BF16 / see protocol |
| TFT_FC | val | local history / weather only | shared across source56 | 0.089270 | 0.138802 | 0.659517 | 0.000452 | 0.016566 | 56 | 10 | 560 | archived BF16 / see protocol |
| TIDE_FC | test | local history / weather only | shared across source56 | 0.085663 | 0.133918 | 0.656729 | 0.000329 | 0.017570 | 56 | 10 | 560 | archived BF16 / see protocol |
| TIDE_FC | val | local history / weather only | shared across source56 | 0.088862 | 0.138045 | 0.663366 | 0.000221 | 0.016189 | 56 | 10 | 560 | archived BF16 / see protocol |
| TIMEXER_FC | test | local history / weather only | shared across source56 | 0.085834 | 0.133894 | 0.656462 | 0.000506 | 0.017317 | 56 | 10 | 560 | archived BF16 / see protocol |
| TIMEXER_FC | val | local history / weather only | shared across source56 | 0.089149 | 0.138234 | 0.662251 | 0.000444 | 0.016150 | 56 | 10 | 560 | archived BF16 / see protocol |
| TRANSFORMER_FC | test | local history / weather only | shared across source56 | 0.084699 | 0.133015 | 0.661216 | 0.000506 | 0.017513 | 56 | 10 | 560 | archived BF16 / see protocol |
| TRANSFORMER_FC | val | local history / weather only | shared across source56 | 0.087732 | 0.136805 | 0.669190 | 0.000364 | 0.016172 | 56 | 10 | 560 | archived BF16 / see protocol |
| VIT_W | test | local history / weather only | shared across source56 | 0.114404 | 0.168766 | 0.410184 | 0.000898 | 0.027675 | 56 | 10 | 560 | archived BF16 / see protocol |
| VIT_W | val | local history / weather only | shared across source56 | 0.121657 | 0.175274 | 0.453308 | 0.000551 | 0.031926 | 56 | 10 | 560 | archived BF16 / see protocol |

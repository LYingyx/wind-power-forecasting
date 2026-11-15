| model | training | context | val_mae | val_seed_sd | val_macro_rmse | val_macro_r2 | test_mae | test_seed_sd | test_macro_rmse | test_macro_r2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Persistence | none | local history / weather only | 0.117542 |  | 0.182200 | 0.421143 | 0.112763 |  | 0.175619 | 0.415666 |
| GRU_H | shared across source56 | local history / weather only | 0.111366 | 0.000035 | 0.169090 | 0.505278 | 0.106572 | 0.000078 | 0.163146 | 0.498368 |
| LSTM_H | shared across source56 | local history / weather only | 0.111398 | 0.000058 | 0.169174 | 0.504921 | 0.106594 | 0.000104 | 0.163259 | 0.497759 |
| PATCHTST_H | shared across source56 | local history / weather only | 0.111493 | 0.000025 | 0.169280 | 0.504068 | 0.106675 | 0.000054 | 0.163309 | 0.497255 |
| ITRANSFORMER_H | shared across source56 | local history / weather only | 0.111667 | 0.000032 | 0.169860 | 0.500715 | 0.106771 | 0.000034 | 0.163770 | 0.494420 |
| MODERNTCN_H | shared across source56 | local history / weather only | 0.111295 | 0.000062 | 0.169113 | 0.505156 | 0.106458 | 0.000085 | 0.163121 | 0.498542 |
| CNN_W | shared across source56 | local history / weather only | 0.122563 | 0.000627 | 0.176882 | 0.443692 | 0.115232 | 0.000379 | 0.170227 | 0.404022 |
| VIT_W | shared across source56 | local history / weather only | 0.121657 | 0.000551 | 0.175274 | 0.453308 | 0.114404 | 0.000898 | 0.168766 | 0.410184 |
| SWIN_W | shared across source56 | local history / weather only | 0.121518 | 0.000529 | 0.175277 | 0.454379 | 0.114129 | 0.000749 | 0.168659 | 0.412168 |
| DLINEAR_FC | shared across source56 | local history / weather only | 0.112123 | 0.000140 | 0.166440 | 0.516924 | 0.108997 | 0.000187 | 0.162071 | 0.501956 |
| NBEATSX_FC | shared across source56 | local history / weather only | 0.089040 | 0.000248 | 0.136094 | 0.673081 | 0.085961 | 0.000360 | 0.132098 | 0.666397 |
| TCN_FC | shared across source56 | local history / weather only | 0.090184 | 0.000228 | 0.140007 | 0.655498 | 0.086435 | 0.000316 | 0.135134 | 0.650038 |
| TFT_FC | shared across source56 | local history / weather only | 0.089270 | 0.000452 | 0.138802 | 0.659517 | 0.085779 | 0.000561 | 0.134527 | 0.653636 |
| TIMEXER_FC | shared across source56 | local history / weather only | 0.089149 | 0.000444 | 0.138234 | 0.662251 | 0.085834 | 0.000506 | 0.133894 | 0.656462 |
| TIDE_FC | shared across source56 | local history / weather only | 0.088862 | 0.000221 | 0.138045 | 0.663366 | 0.085663 | 0.000329 | 0.133918 | 0.656729 |
| TRANSFORMER_FC | shared across source56 | local history / weather only | 0.087732 | 0.000364 | 0.136805 | 0.669190 | 0.084699 | 0.000506 | 0.133015 | 0.661216 |
| CBH_SL | shared across source56 | local history / weather only | 0.084140 | 0.000179 |  |  | 0.081799 | 0.000261 |  |  |
| CBH_IL | independent | local history / weather only | 0.088450 | 0.000129 |  |  | 0.086998 | 0.000111 |  |  |
| CBH_SC | shared across source56 | regional history | 0.083814 | 0.000303 |  |  | 0.081507 | 0.000318 |  |  |
| CBH_IC | independent | regional history | 0.088445 | 0.000101 |  |  | 0.087228 | 0.000116 |  |  |
| E0 | shared across source56 | regional history | 0.083827 | 0.000220 | 0.131481 | 0.692973 | 0.081528 | 0.000242 | 0.128062 | 0.685668 |
| E2C | shared across source56 | regional history | 0.082667 | 0.000129 | 0.129898 | 0.699847 | 0.080703 | 0.000126 | 0.127069 | 0.689828 |

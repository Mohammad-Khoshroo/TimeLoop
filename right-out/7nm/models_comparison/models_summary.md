# Multi-Model Benchmark Summary on ASAP7 7.5T Predictive FinFET

| Model     | Description                               |   Latency_BT2_us |   Latency_BT16_us |   Throughput_BT2_FPS |   Throughput_BT16_FPS |   Speedup_BT16_vs_BT2 |   Energy_per_Frame_uJ |   Core_Power_mW |   Total_Power_mW |
|:----------|:------------------------------------------|-----------------:|------------------:|---------------------:|----------------------:|----------------------:|----------------------:|----------------:|-----------------:|
| RiskMLP   | RiskMLP (Automotive Risk Estimation)      |            4.14  |             4.14  |             241546   |              241546   |                     1 |                 0.171 |           0.541 |            40.38 |
| MNIST-MLP | MNIST-MLP (LeNet-300-100 Benchmark)       |          127.144 |           127.144 |               7865.1 |                7865.1 |                     1 |                 5.163 |           0.527 |            40.57 |
| HAR-MLP   | HAR-MLP (Human Activity Recognition)      |           98.36  |            98.36  |              10166.7 |               10166.7 |                     1 |                 3.996 |           0.527 |            40.58 |
| DLRM-MLP  | DLRM-MLP (Meta Recommendation Bottom MLP) |          129.696 |           129.696 |               7710.3 |                7710.3 |                     1 |                 5.27  |           0.531 |            40.72 |

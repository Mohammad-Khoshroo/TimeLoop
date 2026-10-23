# Multi-Model Benchmark Summary on GlobalFoundries GF180MCU 7T

| Model     | Description                               |   Latency_BT2_us |   Latency_BT16_us |   Throughput_BT2_FPS |   Throughput_BT16_FPS |   Speedup_BT16_vs_BT2 |   Energy_per_Frame_uJ |   Core_Power_mW |   Total_Power_mW |
|:----------|:------------------------------------------|-----------------:|------------------:|---------------------:|----------------------:|----------------------:|----------------------:|----------------:|-----------------:|
| RiskMLP   | RiskMLP (Automotive Risk Estimation)      |            41.4  |             41.4  |              24154.6 |               24154.6 |                     1 |                24.795 |           7.845 |           585.58 |
| MNIST-MLP | MNIST-MLP (LeNet-300-100 Benchmark)       |          1271.44 |           1271.44 |                786.5 |                 786.5 |                     1 |               748.635 |           7.642 |           588.32 |
| HAR-MLP   | HAR-MLP (Human Activity Recognition)      |           983.6  |            983.6  |               1016.7 |                1016.7 |                     1 |               579.42  |           7.645 |           588.39 |
| DLRM-MLP  | DLRM-MLP (Meta Recommendation Bottom MLP) |          1296.96 |           1296.96 |                771   |                 771   |                     1 |               764.15  |           7.695 |           590.47 |

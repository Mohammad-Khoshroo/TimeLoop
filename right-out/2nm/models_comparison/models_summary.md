# Multi-Model Benchmark Summary on GT2N 2nm Nanosheet GAAFET

| Model     | Description                               |   Latency_BT2_us |   Latency_BT16_us |   Throughput_BT2_FPS |   Throughput_BT16_FPS |   Speedup_BT16_vs_BT2 |   Energy_per_Frame_uJ |   Core_Power_mW |   Total_Power_mW |
|:----------|:------------------------------------------|-----------------:|------------------:|---------------------:|----------------------:|----------------------:|----------------------:|----------------:|-----------------:|
| RiskMLP   | RiskMLP (Automotive Risk Estimation)      |            3.45  |             3.45  |             289830   |              289830   |                     1 |                0.065  |           0.246 |            18.42 |
| MNIST-MLP | MNIST-MLP (LeNet-300-100 Benchmark)       |          105.953 |           105.953 |               9438.1 |                9438.1 |                     1 |                1.9619 |           0.24  |            18.5  |
| HAR-MLP   | HAR-MLP (Human Activity Recognition)      |           81.966 |            81.966 |              12200.1 |               12200.1 |                     1 |                1.5185 |           0.24  |            18.5  |
| DLRM-MLP  | DLRM-MLP (Meta Recommendation Bottom MLP) |          108.08  |           108.08  |               9252.4 |                9252.4 |                     1 |                2.0026 |           0.242 |            18.57 |

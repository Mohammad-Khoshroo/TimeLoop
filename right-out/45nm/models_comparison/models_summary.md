# Multi-Model Benchmark Summary on NanGate FreePDK45 Open Cell

| Model     | Description                               |   Latency_BT2_us |   Latency_BT16_us |   Throughput_BT2_FPS |   Throughput_BT16_FPS |   Speedup_BT16_vs_BT2 |   Energy_per_Frame_uJ |   Core_Power_mW |   Total_Power_mW |
|:----------|:------------------------------------------|-----------------:|------------------:|---------------------:|----------------------:|----------------------:|----------------------:|----------------:|-----------------:|
| RiskMLP   | RiskMLP (Automotive Risk Estimation)      |            8.28  |             8.28  |             120773   |              120773   |                     1 |                  1.71 |           2.705 |           201.92 |
| MNIST-MLP | MNIST-MLP (LeNet-300-100 Benchmark)       |          254.288 |           254.288 |               3932.5 |                3932.5 |                     1 |                 51.63 |           2.635 |           202.87 |
| HAR-MLP   | HAR-MLP (Human Activity Recognition)      |          196.72  |           196.72  |               5083.4 |                5083.4 |                     1 |                 39.96 |           2.636 |           202.89 |
| DLRM-MLP  | DLRM-MLP (Meta Recommendation Bottom MLP) |          259.392 |           259.392 |               3855.2 |                3855.2 |                     1 |                 52.7  |           2.653 |           203.61 |

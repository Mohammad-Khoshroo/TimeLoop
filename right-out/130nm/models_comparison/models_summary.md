# Multi-Model Benchmark Summary on SkyWater SKY130 HD

| Model     | Description                               |   Latency_BT2_us |   Latency_BT16_us |   Throughput_BT2_FPS |   Throughput_BT16_FPS |   Speedup_BT16_vs_BT2 |   Energy_per_Frame_uJ |   Core_Power_mW |   Total_Power_mW |
|:----------|:------------------------------------------|-----------------:|------------------:|---------------------:|----------------------:|----------------------:|----------------------:|----------------:|-----------------:|
| RiskMLP   | RiskMLP (Automotive Risk Estimation)      |            20.7  |             20.7  |              48309.2 |               48309.2 |                     1 |                11.628 |           7.359 |           549.23 |
| MNIST-MLP | MNIST-MLP (LeNet-300-100 Benchmark)       |           635.72 |            635.72 |               1573   |                1573   |                     1 |               351.084 |           7.168 |           551.81 |
| HAR-MLP   | HAR-MLP (Human Activity Recognition)      |           491.8  |            491.8  |               2033.3 |                2033.3 |                     1 |               271.728 |           7.17  |           551.87 |
| DLRM-MLP  | DLRM-MLP (Meta Recommendation Bottom MLP) |           648.48 |            648.48 |               1542.1 |                1542.1 |                     1 |               358.36  |           7.217 |           553.82 |

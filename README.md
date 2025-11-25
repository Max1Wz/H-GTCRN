# H-GTCRN
This repository is the un-official implementation of the Interspeech2025 paper: A Lightweight Hybrid Dual Channel Speech Enhancement System under Low-SNR Conditions. For more details, please refer to the [arXiv preprint](https://arxiv.org/abs/2505.19597).

# TODO
dual channel rir generation will finished later.

| ![The framework of our proposed system.](./figures/model.png) |
|:---------------------:|
| **Figure 1:** The framework of our proposed system. |

# Audio samples
The directory structure of the audio samples is shown below:
```markdown
    samples
    ├── Samples1
    |   ├── Samples1_clean.wav
    |   ├── Samples1_noisy.wav
    |   ├── Samples1_IVA.wav
    |   ├── Samples1_GTCRN.wav
    |   ├── Samples1_DC_GTCRN.wav
    |   └── Samples1_Proposed.wav
    | ...
    └── Samples3
        ├── Samples3_clean.wav
        ├── Samples3_noisy.wav
        ├── Samples3_IVA.wav
        ├── Samples3_GTCRN.wav
        ├── Samples3_DC_GTCRN.wav
        └── Samples3_Proposed.wav
```

# Credits
We gratefully acknowledge the following resources that made this project possible:
- [GTCRN](https://github.com/Xiaobin-Rong/gtcrn): SOTA lightweight speech enhancement model architecture.
- [SE-train](https://github.com/Xiaobin-Rong/SEtrain): Excellent training code template for DNN-based speech enhancement.
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
BSS Auxiliary-function-based Indepednent vector analysis
- [AuxIVA](https://github.com/XianruiWang/AuxIVA)

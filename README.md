# Transfer-Learning Benchmark: Oxford-IIIT Pets (24 variants, 12 backbones)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsserGharib1/PetsClassificationML/blob/main/oxford_pets_transfer_learning_benchmark.ipynb)
[![View on nbviewer](https://img.shields.io/badge/view%20full%20notebook-nbviewer-F37626?logo=jupyter&logoColor=white)](https://nbviewer.org/github/AsserGharib1/PetsClassificationML/blob/main/oxford_pets_transfer_learning_benchmark.ipynb)

Systematic benchmark of **12 architectures × 2 regimes** (feature-extraction baseline vs fine-tuned) on the Oxford-IIIT Pet dataset (7,349 images, 37 breeds): VGG16, ResNet-18, WideResNet-50-2, DenseNet121, Xception, Inception-V3, a custom CNN, CapsNet, ViT-B/16, Swin-T, CvT-21, and PoolFormer-S12.

## Top-5 results (test set)

| Rank | Model (fine-tuned) | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|---|
| 1 | **CvT-21** | **95.47%** | 0.9538 | **0.9994** |
| 2 | Swin-T | 94.47% | 0.9438 | 0.9991 |
| 3 | DenseNet121 | 93.83% | 0.9378 | 0.9990 |
| 3 | Inception-V3 | 93.83% | 0.9380 | 0.9987 |
| 5 | ViT-B/16 | 93.29% | 0.9319 | n/a |

Key findings: fine-tuned CvT-21 beat its own feature-extraction baseline by **+5.71 points**, transformer and transformer-hybrid models occupied the entire top-5, consistently ahead of pure CNNs, transfer learning delivered ~95% vs ~70% for comparable training from scratch.

## Benchmark visuals

![Final benchmark comparison](figures/final_benchmark_comparison.png)

![Sample batch](figures/sample_batch.jpg)

## Method

Fixed seeds, one unified 70/15/15 split reused across all 24 variants, shared training/evaluation loops, per-model parameter audits, confusion matrices, learning curves, and multiclass ROC, evaluated on accuracy, macro precision/recall/F1, and ROC-AUC.

## Running

```bash
pip install -r requirements.txt
```
The notebook downloads Oxford-IIIT Pets and reproduces the full grid (checkpoint-resume supported).

# LAM-NET

Local Axial Mamba Network for Multi-class Vertebral Segmentation in CT Images

Authors: Hao Zuo, Dongsheng Zhou, Chao Che, and Tianming Zhao (Dalian University).

## Model implementation

`model.py` contains the PyTorch network definition, including the shared encoder,
Inter-slice Mamba, Adaptive Attention Skip Fusion (AASF), decoder, and segmentation
heads. The class `LAA_Mamba_SpineSeg` implements LAM-NET; `LAA` implements AASF.

The model requires PyTorch. Its Mamba backend uses `mamba-ssm` and
`causal-conv1d`; the included S4D fallback is a different backend and does not
reproduce the paper's Mamba experiment. Install dependencies from their upstream
projects in a compatible CUDA environment.

The model returns a dictionary with `logits_25`, `logits_4`, and `logits_2`.
Input organization is controlled by the constructor arguments; inter-slice inputs
have shape `(B, K, C, H, W)`.

## Scope

This repository provides the model implementation. Data loading and preprocessing,
training and checkpoint selection, experiment split files, evaluation scripts,
and trained weights are not included. The model file alone is not an end-to-end
reproduction package. Dataset access is subject to the original providers' terms.

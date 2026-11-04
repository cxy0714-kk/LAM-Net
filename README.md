# LAM-NET

Local Axial Mamba Network for Multi-class Vertebral Segmentation in CT Images

Authors: Hao Zuo, Dongsheng Zhou, Chao Che, and Tianming Zhao (Dalian University).

## Current release scope

The current `main` branch is a **limited, documentation-only interface preview**.
It is not a runnable implementation and cannot by itself reproduce the paper's
training or evaluation. `model.py` contains only an explicitly non-executable
interface stub; it is not the network used for the reported experiments.

The following are **not included in the current tree**:

- Network internals, including Inter-slice Mamba, AASF, and the decoder.
- Data loading, preprocessing, label mapping, and the experiment split.
- Training, checkpoint selection, and physical-distance evaluation routines.
- Training configurations, trained weights, CT images, annotations, and logs.

## Method overview

LAM-NET combines sampled axial contextual conditioning, adaptive attention skip
fusion, and hierarchical auxiliary supervision for vertebral segmentation in CT.
Consult the manuscript for the method and experimental protocol. This preview
does not provide commands for reproducing the reported numerical results.

## Version history

This scope describes the latest branch contents. Earlier commits contained a
more complete source release; replacing the current files does not remove those
commits or copies already obtained by others. The implementation is not bundled
with this current preview. A future full release, if made, will be identified
explicitly here.

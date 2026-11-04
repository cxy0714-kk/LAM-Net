# LAM-NET

Local Axial Mamba Network for Multi-class Vertebral Segmentation in CT Images

Authors: Hao Zuo, Dongsheng Zhou, Chao Che, and Tianming Zhao (Dalian University).

## Public source code

`model.py` is a self-contained source release containing the frozen main model,
foreground/coarse-region label mapping, volume-level training pipeline, the
337/42/42 paper split (`PAPER_SPLIT`), archived settings (`MAIN_CONFIG`), and
physical-distance evaluation. The historical class name `LAA_Mamba_SpineSeg`
corresponds to LAM-NET; class `LAA` implements AASF.

The model classes are computationally unchanged from the frozen experiment.
Public packaging consolidates modules, makes data paths configurable, embeds the
fixed split/configuration, and aligns the training default learning rate to 3e-5.
The frozen implementation retains historical modules and parameters; its forward
path is selected by the paper configuration. The main model has 28,060,478 parameters.

## Environment

The recorded experiment used Python 3.8, PyTorch 2.4.1, mamba-ssm 2.2.2,
causal-conv1d 1.4.0 and an RTX 3090. Install a compatible CUDA/PyTorch environment
on Linux, plus NumPy, SciPy 1.10.1, nibabel 5.2.1, matplotlib 3.7.5, einops 0.8.1,
and ninja. Install compatible builds of mamba-ssm 2.2.2 and causal-conv1d 1.4.0
from their upstream projects; their respective licenses apply. The S4D fallback
is a different backend and does not reproduce the paper's Mamba run.

## Data

Obtain CTSpine1K from its original providers and follow their terms. No images,
annotations, trained weights or private research logs are redistributed here.
`PAPER_SPLIT` contains public dataset file identifiers only.

Set `LAMNET_DATA_ROOT` to a directory with these subdirectories:

```
HNSCC-3DCT-RT/volume/ and HNSCC-3DCT-RT/label/
CTSpineMSD/volume/ and CTSpineMSD/label/
COVID-19/volume/ and COVID-19/label/
COLONOG/volume/ and COLONOG/label/
```

HNSCC/MSD/COLONOG labels use `_seg.nii.gz`; COVID labels use `_ct_seg.nii.gz`.
The loader selects 15 fixed positions per volume and resizes slices to 256x256.
K counts sampled positions, not necessarily adjacent original CT slices.

## Training and evaluation

```bash
export LAMNET_DATA_ROOT=/path/to/data
python model.py --smoke-test
python model.py --generate_split_only --seed 42
python model.py --seed 42 --split_seed 42 --epochs 100 --batch_size 8 \
  --lr 3e-5 --lam2 0.3 --lam4 0.3 --use_3slice_input --use_zpos \
  --use_interslice_mamba --interslice_K 5 --output_dir runs/Full_V17
python model.py --physical-eval --checkpoint runs/Full_V17/seed_42/best.pth --output test_results.json
```

Checkpoint selection uses validation MoV Dice. The embedded seed-42 split must
remain volume-disjoint. The release smoke check verifies real Mamba, output shapes,
finite logits, 28.06 M parameters, spacing conversion and empty-mask handling.
A full training reproduction was not rerun as part of packaging the release.

MoV Dice/mIoU average ground-truth-present classes within each target slice,
then average slices. HD95/ASD are binary-foreground, 2D metrics. The original
training log distances are in pixels; use `--physical-eval` for mm. Effective
pixel spacing is `(NIfTI_spacing_axis0 * original_H / 256,
NIfTI_spacing_axis1 * original_W / 256)`. ASD equally averages the two directional
surface-to-opposite-foreground means, following the archived protocol. An empty
predicted/reference foreground gives no finite distance and is excluded from
distance means; valid counts are reported. These conventions differ from some
surface-only and full-volume metrics.

Third-party baseline code/checkpoints are obtained separately from their authors.

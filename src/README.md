# src/

Source code for the 3D fracture classification, segmentation, and restoration
pipeline.

---

## Directory Layout

```
src/
  data/               Dataset classes (PyTorch and TensorFlow)
  training/           Training pipelines
    pointnet_cls/     PointNet binary classifier (TensorFlow / Keras)
  pointnext/          PointNeXt framework (git submodule, forked openpoints)
  ptv3_setup/         Pointcept dataset adapters and config files for this project
```

---

## Module Descriptions

### `data/`

| File | Purpose |
|---|---|
| `dataset_pytorch.py` | PyTorch `Dataset` for loading `.pts`/`.seg` segmentation files. Returns `{points, labels, object_id}` dicts. Supports fixed-size point sampling and normalization. For future segmentation training. |
| `dataset_tensorflow.py` | TensorFlow `tf.data.Dataset` equivalent of the PyTorch dataset. Mirrors the same loading / sampling / normalization logic. |

### `training/pointnet_cls/`

Full PointNet binary classifier (complete vs broken) trained on Fantastic Breaks
with TensorFlow/Keras. The last experiment used 2048 points and the 11D geometric
feature set.

| File | Purpose |
|---|---|
| `model.py` | PointNet model architecture (classification head) |
| `train.py` | Training entry point |
| `evaluate.py` | Evaluation script (metrics, confusion matrix, inference samples) |
| `configs/default.py` | Default hyperparameter config |
| `utils/` | Training helpers: `data.py`, `training.py`, `evaluation.py`, `metrics.py`, `augmentations.py`, `inference.py`, `plotting.py`, `io.py`, `compat.py` |

### `pointnext/`

Forked [PointNeXt / OpenPoints](https://github.com/guochengqian/PointNeXt) framework
(git submodule at `src/pointnext`). Contains the PointNeXt-B classification training
loop, configs under `cfgs/`, and experiment logs.

### `ptv3/`

Cloned [Pointcept](https://github.com/Pointcept/PointTransformerV3) framework for
Point Transformer V3 experiments. Contains the full Pointcept training infrastructure.

### `ptv3_setup/`

Project-specific Pointcept integration files:

| File | Purpose |
|---|---|
| `fantasticbreaks.py` | Pointcept dataset adapter for Fantastic Breaks (HDF5 reader, registered as `FantasticBreaksClsDataset`) |
| `breakingbad.py` | Pointcept dataset adapter for Breaking Bad (HDF5 reader, registered as `BreakingBadClsDataset`) |
| `cls-ptv3-v1m1-0-base.py` | PTv3-Base classification config for Fantastic Breaks |
| `cls-ptv3-v1m1-0-small.py` | PTv3-Small classification config for Fantastic Breaks |
| `cls-ptv3-bb-v1m1-0-small.py` | PTv3-Small classification config for Breaking Bad |

To use these with Pointcept:
1. Copy `fantasticbreaks.py` / `breakingbad.py` to `src/ptv3/pointcept/datasets/`
2. Register them in `src/ptv3/pointcept/datasets/__init__.py`
3. Copy the config `.py` files to `src/ptv3/configs/<experiment>/`

---

## Experiment Results

Experiment outputs (checkpoints, logs, plots) are stored in `results/` at the
repository root — **not** inside `src/`.

```
results/
  pointnet_cls/     PointNet classifier experiment outputs
```

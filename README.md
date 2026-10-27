# FARP-Seg

FARP-Seg converts three-dimensional membrane prediction volumes into cell instance segmentations. It combines region filling, learned agglomeration, risk-aware repartitioning, fragment stitching, and morphological cleanup in a desktop application.

The interface supports English and Chinese, synchronized prediction and segmentation views, and adjustable processing parameters. Inference runs locally on the CPU.

## Installation

Use Python 3.11. The desktop application has been tested on Windows.

From the repository root, install the dependencies:

```bash
python -m pip install -r software/requirements.txt
```

Start the application:

```bash
cd software
python app.py
```

On Windows, `software/start.bat` also starts the application.

## Models and example data

Model weights, calibration parameters, and morphology libraries are stored under `software/resources/models/`. The model collection contains three folds. The supplied example automatically selects its associated fold and recommended parameters.

```text
software/
├── app.py
├── requirements.txt
└── resources/
    ├── models/
    │   ├── a14_fold0.txt
    │   ├── a14_fold1.txt
    │   ├── a14_fold2.txt
    │   ├── manifest.json
    │   └── full/
    │       ├── assets.json
    │       ├── fold0/
    │       ├── fold1/
    │       └── fold2/
    └── sample/
        ├── 771_2x_prediction.tiff
        ├── 771_2x_prediction.json
        └── default.json
```

The example is **`1-I_1-II/771_2x`**, a prediction volume with dimensions **181 × 1016 × 1016** in Z × Y × X order. Its recommended configuration uses model fold 1, a merge threshold of 0.972, and XY sampling by a factor of 2.

## Quick start

1. Click **Select sample** and choose `771_2x`. To use your own data, click **Import folder** and select a folder containing prediction volumes.
2. Click **Run segmentation**. Recommended settings are applied automatically to the supplied example.
3. Inspect the result using instance colors, boundary overlay, or blended overlay. The prediction and segmentation views share the same slice, zoom, and pan.
4. Click **Export segmentation** to save the label volume.

During processing, the corresponding run button becomes **Cancel segmentation** or **Cancel preprocessing**. Cancelling stops the task and retains any previously completed result.

## Input

The input is a single-channel membrane prediction volume in **Z × Y × X** order. Supported formats are TIFF (`.tif`, `.tiff`), NumPy (`.npy`, `.npz`), and HDF5 (`.h5`, `.hdf5`). Two-dimensional inputs are treated as single-slice volumes.

Folders are scanned for available samples. A prediction file can also be dragged into the application. For HDF5 files with multiple candidate datasets, specify the dataset path in **Advanced settings**.

## Parameters and preprocessing

The main panel exposes two cleanup controls:

| Parameter | Default |
| --- | --- |
| Background cleanup strength | 0.10 |
| Edge cleanup strength | 6 |

Model selection, merge threshold, XY sampling, processing switches, and thread settings are available under **Advanced settings**. **Reset defaults** restores the recommended configuration.

Optional preprocessing enhances blurry or dim membrane predictions. It is disabled by default and runs as a separate step: load a sample, click **Run preprocessing**, then run segmentation using the preprocessed input. Its worker count is adjustable in **Advanced settings**.

Processing time and memory use depend on the volume dimensions and selected settings. A low-memory estimate produces a warning and processing continues; an allocation failure stops the task while keeping the application available.

## Output

Each segmentation run is saved in a separate directory under `software/outputs/`:

- `segment.npy`: the final instance label volume.
- `segment.json`: processing parameters, model information, and execution metadata.

Export supports TIFF, NumPy, and HDF5. Labels use unsigned 32-bit integers, with **0** representing background. Exported volumes retain the input dimensions; when XY sampling is enabled, labels are restored to the input grid using nearest-neighbor expansion. Each export includes a JSON metadata file.

# RT-NET
RT-Net: A Deep Learning-Based Emulator Bridging 1D and 3D Radiative Transfer Models for Efficient and Accurate Canopy BRF Simulation.

## 1. What's RT-Net?

RT-Net learns the BRF difference between the 1D radiative transfer model 4SAIL and the 3D model LESS. For a given reference optical-property condition, it predicts a scalar correction to the 4SAIL BRF. The corrected BRFs at 56 predefined reference conditions can then be used by Semi-LESS to reconstruct BRF across 400–2500 nm.

This repository contains one model for homogeneous canopies and five models for row-structured canopies with different leaf angle distributions (LADs):

| Model file | Canopy scenario | Input variables |
| --- | --- | ---: |
| `RT-NET-homogeneous.h5` | Homogeneous | 12 |
| `RT-NET-extremophile.h5` | Row-structured, extremophile LAD | 15 |
| `RT-NET-plagiophile.h5` | Row-structured, plagiophile LAD | 15 |
| `RT-NET-planophile.h5` | Row-structured, planophile LAD | 15 |
| `RT-NET-spherical.h5` | Row-structured, spherical LAD | 15 |
| `RT-NET-uniform.h5` | Row-structured, uniform LAD | 15 |

Each model produces one scalar output per input row.

## 2. Model inputs

`test-input.txt` is a comma-separated test file that can be supplied directly to Row-structured .h5 model, but it just a test file. Its **15 columns must remain in this order**:

| Position | Column | Description |
| ---: | --- | --- |
| 1 | `LAI` | Leaf area index |
| 2 | `HSPOT` | Hotspot parameter |
| 3 | `Row Width` | Row width |
| 4 | `Plant Height` | Plant height |
| 5 | `Ridge Spacing` | Ridge spacing |
| 6–7 | `sin(TTS)`, `cos(TTS)` | Sine and cosine of the solar zenith angle |
| 8–9 | `sin(TTO)`, `cos(TTO)` | Sine and cosine of the viewing zenith angle |
| 10–11 | `sin(VAA)`, `cos(VAA)` | Sine and cosine of the azimuth angle used in training |
| 12–13 | `LIDFA`, `LIDFB` | Leaf angle distribution parameters |
| 14 | `Wavelength(nm)` | Actual wavelength (nm) corresponding to the Semi-LESS feature band |
| 15 | `BRF` | Corresponding PROSAIL input BRF |

The homogeneous model accepts 12 inputs **(without Row Width, Plant Height and Ridge Spacing)**; the row models accept all 15 inputs.

## 3. Quick start

Model loading and inference have been tested with Python 3.12 and TensorFlow 2.16.2.

```bash
pip install tensorflow-cpu==2.16.2 numpy
```

Place `.h5` in `models/` and `test-input.txt` in `data/`, then run:

```python
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(
    "models/.h5",
    compile=False,
)

inputs = np.loadtxt(
    "data/test-input.txt",
    delimiter=",",
    skiprows=1,
    dtype=np.float32,
)

assert inputs.shape[1] == model.input_shape[-1] == 15

predictions = model.predict(inputs, batch_size=4096, verbose=0)

print(f"Input shape:  {inputs.shape}")
print(f"Output shape: {predictions.shape}")
print(f"First five predictions: {predictions[:5, 0]}")

np.savetxt(
    "predictions.csv",
    predictions,
    delimiter=",",
    header="predicted_correction",
    comments="",
)
```

This example reads the test file as provided and saves one prediction per input row. Load the models with `compile=False` for inference.

## Citation

[RT-Net: A Deep Learning-Based Emulator Bridging 1D and 3D Radiative Transfer Models for Efficient and Accurate Canopy BRF Simulation]

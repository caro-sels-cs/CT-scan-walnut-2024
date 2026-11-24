# CT Scan Reconstruction (Walnut)

Python implementation of a Computed Tomography (CT) image reconstruction pipeline applied to a divergent (fan-beam) sinogram of a walnut.

## Overview
The script reconstructs a 2D cross-sectional image from raw X-ray projection data (`walnut_divergent.png`) using the following steps:
1. **Fan-to-Parallel Rebinning:** Converts divergent fan-beam projection geometry into parallel-beam sinogram coordinates.
2. **Frequency Filtering:** Applies a 1D Fourier transform and a **Ram-Lak (ramp) filter** to sharpen edges and reduce blurring during backprojection.
3. **Filtered Backprojection (FBP):** Rotates and accumulates the filtered projections across 360 degrees to reconstruct the final image.

## Files
* `ct_reconstruction.py` – Main script containing the rebinning, Ram-Lak filter, and Radon backprojection functions.
* `walnut_divergent.png` – Input sinogram dataset.

## Usage
1. Install the required dependencies:
   ```bash
   pip install numpy opencv-python matplotlib scikit-image

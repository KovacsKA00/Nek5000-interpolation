# Interpolation of Nek5000 Output to Equidistant VTK Format

## Overview
Interpolation script for a three-dimensional Kolmogorov flow DNS database.

This script interpolates velocity field data from Nek5000 output files to a uniform equidistant grid, and saves the results in `.vtk` format for post-processing and visualization (e.g., with ParaView).

## Dependencies
Make sure the following Python packages are installed:

```bash
pip install os pymech numpy fnmatch numba vtk pyvista
```

## Usage
1. Place the script in the root directory above the `DNS_Kolmogorov` folder.
2. Run the script:
   ```bash
   python interpolation.py
   ```
3. For each `.f0XXXX` file in the case directories, a corresponding `vel_XXXX.vtk` file will be saved with interpolated velocity fields.

## Notes
1. When interpolating an originally $128^3$ grid, if `double_yn` is set to `True`, then `n = 256` should be written.
2. When interpolating an originally $256^3$ grid, if `double_yn` is set to `True`, then `n = 512` should be written.

## Output
Each generated `.vtk` file will contain:
1. A structured grid with dimensions `n x n x n`.
2. A `Velocity` vector field (3D) at each grid point.

The `.vtk` files can be loaded in ParaView or other VTK-compatible tools for visualization and analysis.

Tested with Python 3.10.11 and ParaView 5.13.3.

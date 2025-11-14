# ALMA Data Cleaning Tools

General tools for ALMA data cleaning and visualization.

## Features

- **Channel Map Plotting**: Visualize channel maps with optional contour overlays
  - Keplerian mask contours (from FITS files)
  - CARTA-style sigma contours (3σ, 5σ, etc.)
  - CARTA-style Gaussian smoothing before contouring

## Installation

```bash
cd cleaning-tools
pip install -e .
```

## Usage

See [examples/channel_maps_demo.ipynb](examples/channel_maps_demo.ipynb) for detailed usage examples.

```python
from cleaning_tools import ChannelMapPlotter

# Create plotter instance
plotter = ChannelMapPlotter(
    cube_path='path/to/cube.fits',
    keplerian_mask_path='path/to/mask.fits'  # optional
)

# Plot channel maps with sigma contours
plotter.plot_channel_maps(
    start_channel=20,
    end_channel=50,
    sigma_levels=[3, 5, 10],
    smooth_sigma=1.7,  # CARTA-style smoothing
    channels_per_page=20
)
```

## Dependencies

- numpy
- matplotlib
- astropy
- scipy
- bettermoments

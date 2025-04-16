# CitySeg Demos

This directory contains demo scripts and notebooks that showcase how to use CitySeg.

## Available Demos

- `component_demo.ipynb` - Demonstrates the component-based architecture introduced in CitySeg 0.4.0

## Example Inputs

The `example_inputs/` directory contains sample images and videos for use with the demos:

- `EustonTap-Screenshot1.png` - Sample street scene image
- `CaledonianPark1_15s_3840x2160.mov` - Sample street scene video (15 seconds)

## Running the Demos

The Python scripts can be run directly:

```bash
cd docs/demos
python component_demo.ipynb
```

Or converted to Jupyter notebooks using [jupytext](https://github.com/mwouts/jupytext):

```bash
pip install jupytext
jupytext --to notebook component_demo.ipynb
jupyter notebook component_demo.ipynb
```

## Note on Output Directory

The demos will create an `outputs/` directory to store generated visualizations and results.
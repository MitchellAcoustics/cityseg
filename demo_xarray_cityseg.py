"""Demo of CitySeg xarray-based architecture.

This script demonstrates the key functionality of the new xarray-based CitySeg:
1. Creating synthetic segmentation data
2. Using the .seg accessor for analysis
3. Loading real models for segmentation
4. Saving and loading datasets
"""
# %%

import tempfile
from pathlib import Path

import cityseg

# %%
"""Demonstrate real segmentation with models."""
print("\\n=== Real Segmentation Demo ===")

# Create a simple test image
test_image = (
    "/Users/mitch/Documents/GitHub/cityseg/example_inputs/EustonTap-Screenshot1.png"
)

print("Loading segmentation model...")
model, processor = cityseg.load_segmentation_model()

print("Running segmentation on test image...")
ds = cityseg.segment_image(
    image=test_image, model=model, processor=processor, return_confidence=True
)

print(f"Segmentation result: {ds.sizes}")
print(f"Data variables: {list(ds.data_vars)}")
print(f"Model used: {ds.attrs['model_name']}")
print(f"Number of classes: {ds.seg.num_classes}")

# Show class distribution
stats = ds.seg.class_stats()
print("\\nTop 5 classes by area:")
top_classes = stats.nlargest(5, "pixel_count_sum")[["class_name", "percentage_mean"]]
for _, row in top_classes.iterrows():
    print(f"  {row['class_name']}: {row['percentage_mean']:.1f}%")


# %%

# Save in different formats
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)

    # Save as Zarr
    zarr_path = tmp_path / "demo.zarr"
    cityseg.save_segmentation_dataset(ds, zarr_path, format="zarr")
    print(f"Saved as Zarr: {zarr_path}")

    # Save as NetCDF
    nc_path = tmp_path / "demo.nc"
    cityseg.save_segmentation_dataset(ds, nc_path, format="netcdf")
    print(f"Saved as NetCDF: {nc_path}")

    # Load back
    loaded_ds = cityseg.load_segmentation_dataset(zarr_path)
    print(f"Loaded dataset: {loaded_ds.sizes}")
    print(f"Attributes preserved: {list(loaded_ds.attrs.keys())}")

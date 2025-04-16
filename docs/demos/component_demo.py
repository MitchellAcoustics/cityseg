#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# CitySeg Component-Based Architecture Demo

This notebook demonstrates how to use the CitySeg component-based architecture
introduced in version 0.4.0. It shows both the legacy API for backward compatibility
and the new modular component approach for more flexibility.

Note: This script can be converted to a Jupyter notebook using:
```
jupytext --to notebook component_demo.py
```
"""

# %% [markdown]
# ## Setup and Imports

# %%
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import CitySeg
import cityseg as cs
from cityseg.core.config import Config, ModelConfig, VisualizationConfig
from cityseg.components import ImageProcessor, SegmentationProcessor
from cityseg.analysis import VisualizationHandler

# Set paths
# Use relative paths for demo
EXAMPLE_DIR = Path("example_inputs")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ## Configuration
#
# First, we'll set up a basic configuration for our image processing pipeline.

# %%
example_image = EXAMPLE_DIR / "EustonTap-Screenshot1.png"
example_video = EXAMPLE_DIR / "CaledonianPark1_15s_3840x2160.mov"

# Check if files exist
print(f"Image exists: {example_image.exists()}")
print(f"Video exists: {example_video.exists()}")

# Create a simple configuration
model_config = ModelConfig(
    name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    model_type="segformer",
    device="mps",  # Use "cuda" for GPU if available
    max_size=640,  # Resize to this maximum dimension
    num_workers=0,  # No multiprocessing for demo
)

vis_config = VisualizationConfig(
    alpha=0.7,  # 70% opacity for overlay
    colormap="default",  # Use default colormap for the model
)

config = Config(
    input=example_image,
    output_dir=OUTPUT_DIR,
    output_prefix="demo",
    model=model_config,
    visualization=vis_config,
    frame_step=5,  # Only used for video
    save_raw_segmentation=True,
    save_colored_segmentation=True,
    save_overlay=True,
    analyze_results=True,
    disable_tqdm=False,  # Show progress bars
)

# Print the configuration
print(f"Configuration created for {config.input_type.name}")

# %% [markdown]
# ## Method 1: Legacy Interface
#
# CitySeg maintains backward compatibility with the previous processor-based API.

# %%
# Create a processor for the input (simplest approach)
processor = cs.create_processor(config)

# Process the input (commented out to avoid executing in demo)
# processor.process()

print("Legacy processor created (execution commented out for demo)")
print(f"Processor type: {type(processor).__name__}")

# %% [markdown]
# ## Method 2: Component-Based Approach
#
# The new component-based architecture provides more flexibility and control.

# %% [markdown]
# ### Image Segmentation with Components

# %%
# Load the image
image = ImageProcessor.load_image(example_image)
print(f"Image loaded: {type(image)}, size: {image.size}")

# Resize for processing (keeping aspect ratio)
resized_image = ImageProcessor.resize_image(image, model_config.max_size)
print(f"Resized to: {resized_image.size}")

# Display the image
plt.figure(figsize=(10, 6))
plt.imshow(resized_image)
plt.title("Original Image")
plt.axis("off")
plt.show()

# %% [markdown]
# ### Create and Apply Segmentation Model

# %%
# Create the segmentation pipeline
segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)
print(f"Segmentation pipeline created: {type(segmentation_pipeline).__name__}")

# Process the image
result = SegmentationProcessor.process_image(resized_image, segmentation_pipeline)

# Extract segmentation map and metadata
seg_map = result["seg_map"]
palette = result.get("palette")
id2label = result.get("id2label", {})

print(f"Segmentation map shape: {seg_map.shape}")
print(f"Number of classes in result: {len(id2label)}")

# Show a few class labels as example
sample_labels = list(id2label.items())[:5]
print("Sample class labels:")
for class_id, class_name in sample_labels:
    print(f"  - ID {class_id}: {class_name}")

# %% [markdown]
# ### Visualize Segmentation Results

# %%
# Create colored segmentation map
colored_segmentation = VisualizationHandler.visualize_segmentation(
    np.array(resized_image), seg_map, palette, colored_only=True
)

# Create overlay visualization
overlay = VisualizationHandler.visualize_segmentation(
    np.array(resized_image), seg_map, palette, colored_only=False, alpha=0.6
)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(resized_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(colored_segmentation)
plt.title("Segmentation Map")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlay Visualization")
plt.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analyze Segmentation Results

# %%
# Get unique classes and count pixels per class
unique_classes, counts = np.unique(seg_map, return_counts=True)
total_pixels = seg_map.size

# Create a list of class information
class_info = []
for cls_id, count in zip(unique_classes, counts):
    class_name = id2label.get(int(cls_id), f"Unknown-{cls_id}")
    percentage = 100 * count / total_pixels
    class_info.append(
        {
            "id": int(cls_id),
            "name": class_name,
            "pixels": int(count),
            "percentage": float(percentage),
        }
    )

# Sort by percentage (descending)
class_info.sort(key=lambda x: x["percentage"], reverse=True)

# Display analysis
print("Segmentation Analysis:")
for cls in class_info:
    print(
        f"  - {cls['name']} ({cls['id']}): {cls['percentage']:.2f}% ({cls['pixels']} pixels)"
    )

# %% [markdown]
# ### Save Results
#
# CitySeg provides built-in functionality for saving results, including both visualizations
# and structured data storage using Zarr and Parquet.

# %%
# 1. Save the visualizations
output_image = OUTPUT_DIR / "demo_original.png"
output_segmentation = OUTPUT_DIR / "demo_segmentation.png"
output_overlay = OUTPUT_DIR / "demo_overlay.png"

Image.fromarray(np.array(resized_image)).save(output_image)
Image.fromarray(colored_segmentation).save(output_segmentation)
Image.fromarray(overlay).save(output_overlay)

print(f"Visualizations saved to {OUTPUT_DIR}:")
print(f"  - Original: {output_image}")
print(f"  - Segmentation: {output_segmentation}")
print(f"  - Overlay: {output_overlay}")

# 2. Save segmentation data and analysis using CitySeg's storage components
from cityseg.analysis import SegmentationAnalyzer  # noqa: E402
import xarray as xr  # noqa: E402

# Create dataset for a single frame (for video, we'd have multiple frames)
# First create a data array with time dimension for consistency
seg_data = xr.DataArray(
    # Add a time dimension for consistency with video format
    np.expand_dims(seg_map, axis=0),
    dims=["time", "y", "x"],
    coords={
        "time": [0],  # Single frame
        "y": np.arange(seg_map.shape[0]),
        "x": np.arange(seg_map.shape[1]),
    },
)

# Create an xarray Dataset with the segmentation data
dataset = xr.Dataset({"segmentation": seg_data})

# Add metadata as attributes
dataset.attrs = {
    "model": model_config.name,
    "model_type": model_config.model_type,
    "frame_step": 1,
    "input_file": str(example_image),
    "original_size": f"{image.width}x{image.height}",
    "id2label": id2label,  # Model's class mapping
}

# Save segmentation data to Zarr and analysis to Parquet using CitySeg's analyzer
zarr_output_path = OUTPUT_DIR / "demo_segmentation"
analysis_output_path = SegmentationAnalyzer.analyze_segmentation_dataset(
    dataset, zarr_output_path
)

print("\nStructured data saved:")
print(f"  - Segmentation data (Zarr): {zarr_output_path}.zarr")
print(f"  - Analysis results (Parquet): {analysis_output_path}")

# Read back the saved analysis data to display some statistics
import pandas as pd  # noqa: E402

analysis_df = pd.read_parquet(analysis_output_path)

print("\nAnalysis summary (top categories by percentage):")
top_categories = analysis_df.sort_values("percentage", ascending=False).head(3)
for _, row in top_categories.iterrows():
    category_id = row["category_id"]
    category_name = id2label.get(category_id, f"Unknown-{category_id}")
    print(
        f"  - {category_name}: {row['percentage']:.2f}% ({row['pixel_count']} pixels)"
    )

# %% [markdown]
# ## Video Processing with Components
#
# The component architecture also supports video processing. Here's a basic example
# (commented out as execution would take longer).

# %%
# Using the video components for frame extraction

from cityseg.components import VideoProcessor  # noqa: E402

# Get video metadata
metadata = VideoProcessor.get_metadata(example_video)
print(f"Video metadata: {metadata}")

# Extract specific frames (e.g., first 3 frames at specified intervals)
frame_indices = [0, 30, 60]
frames = VideoProcessor.get_frames(example_video, frame_indices)
print(f"Extracted {len(frames)} frames")

# Process each frame
processed_frames = []
for i, frame in enumerate(frames):
    # Resize frame
    resized_frame = ImageProcessor.resize_image(frame, model_config.max_size)

    # Process with segmentation model
    result = SegmentationProcessor.process_image(resized_frame, segmentation_pipeline)

    # Create overlay
    overlay = VisualizationHandler.visualize_segmentation(
        np.array(resized_frame),
        result["seg_map"],
        result.get("palette"),
        colored_only=False,
        alpha=0.6,
    )

    # Save frame
    output_path = OUTPUT_DIR / f"frame_{i}_overlay.png"
    Image.fromarray(overlay).save(output_path)
    processed_frames.append(overlay)

    print(f"Processed frame {i}, saved to {output_path}")


# %% [markdown]
# ## Using the Hamilton Workflow
#
# CitySeg also provides a high-level workflow based on Hamilton for automated caching and dependency tracking.

# %%
"""
# Process with Hamilton workflow
from cityseg.workflow import process

# This will automatically handle caching and dependency tracking
result = process(config)
print(f"Workflow completed, result type: {type(result)}")

# The workflow result contains all outputs from the pipeline
print("Available keys in result:")
print(list(result.keys()))
"""
print("Hamilton workflow example code is commented out to avoid execution time")

# %% [markdown]
# ## Conclusion
#
# The new component-based architecture in CitySeg 0.4.0 provides:
#
# - More granular control over each step of the pipeline
# - Better separation of concerns between components
# - Flexible composition of components for custom workflows
# - Backward compatibility through the legacy interface
#
# While maintaining all the functionality of the original CitySeg library.

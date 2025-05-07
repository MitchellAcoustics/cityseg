# Image Processing Guide

This guide explains how to use CitySeg for processing images, generating segmentation maps, and analyzing the results.

## Basic Image Processing

Processing a single image with CitySeg is straightforward. You can use either the legacy interface or the component-based architecture.

### Using the Legacy Interface

The simplest way to process an image is with the legacy interface:

```python
import cityseg as cs

# Create or load configuration
config = cs.Config(
    input="path/to/your/image.jpg",
    output_dir="results",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        device="auto"  # Automatically select best device
    )
)

# Create processor and process the image
cs.create_processor(config).process()
```

### Using the Component Architecture

For more control over the processing pipeline, use the component-based architecture:

```python
import cityseg as cs
from cityseg.components import ImageProcessor, SegmentationProcessor
from cityseg.analysis import VisualizationHandler
from cityseg.storage import StorageAdapter

# Create or load configuration
config = cs.Config.from_yaml("config.yaml")

# Load and process image with fine-grained control
image = ImageProcessor.load_image(config.input)
resized_image = ImageProcessor.resize_image(image, config.model.max_size)

# Create segmentation pipeline and process image
segmentation_pipeline = SegmentationProcessor.create_pipeline(config.model)
result = SegmentationProcessor.process_image(resized_image, segmentation_pipeline)

# Extract segmentation map and metadata
seg_map = result["seg_map"]
palette = result.get("palette")
id2label = result.get("id2label", {})

# Create visualizations
colored_segmentation = VisualizationHandler.visualize_segmentation(
    image, seg_map, palette, colored_only=True
)
overlay = VisualizationHandler.visualize_segmentation(
    image, seg_map, palette, colored_only=False, alpha=config.visualization.alpha
)

# Save results
from PIL import Image
import numpy as np

# Save visualizations
Image.fromarray(overlay).save("results/segmentation_overlay.png")
Image.fromarray(colored_segmentation).save("results/colored_segmentation.png")

# Save raw segmentation data for later analysis
storage = StorageAdapter()
output_path = storage.save_segmentation_data(
    seg_map, {"id2label": id2label}, "results/segmentation_data"
)
```

## Processing a Batch of Images

To process multiple images efficiently, you can either:

1. Point to a directory containing images
2. Process images individually in a batch

### Processing a Directory of Images

```python
import cityseg as cs

# Configure for a directory of images
config = cs.Config(
    input="path/to/image_directory",  # Directory containing images
    output_dir="results",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    )
)

# Process all images in the directory
cs.create_processor(config).process()
```

### Processing Images in a Custom Batch

```python
import cityseg as cs
from cityseg.components import ImageProcessor, SegmentationProcessor
from cityseg.analysis import VisualizationHandler
from pathlib import Path

# Create configuration for model
model_config = cs.ModelConfig(
    name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    device="auto"
)

# Create segmentation pipeline once for all images
segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)

# Process multiple images
image_paths = list(Path("path/to/images").glob("*.jpg"))

for image_path in image_paths:
    # Load and process image
    image = ImageProcessor.load_image(image_path)
    resized_image = ImageProcessor.resize_image(image, model_config.max_size)
    
    # Process image
    result = SegmentationProcessor.process_image(resized_image, segmentation_pipeline)
    
    # Create visualization
    overlay = VisualizationHandler.visualize_segmentation(
        image, result["seg_map"], result.get("palette"), alpha=0.6
    )
    
    # Save result
    output_path = Path("results") / f"{image_path.stem}_overlay.png"
    ImageProcessor.save_image(overlay, output_path)
```

## Analyzing Segmentation Results

After processing an image, you can analyze the segmentation results to extract meaningful insights.

### Basic Segmentation Analysis

```python
import cityseg as cs
import numpy as np

# Get segmentation result (assuming you've already processed an image)
result = ...  # Output from SegmentationProcessor.process_image
seg_map = result["seg_map"]
id2label = result.get("id2label", {})

# Count pixels per class
unique_classes, counts = np.unique(seg_map, return_counts=True)
total_pixels = seg_map.size

# Display class distribution
print("Class distribution:")
for cls_id, count in zip(unique_classes, counts):
    class_name = id2label.get(int(cls_id), f"Unknown-{cls_id}")
    percentage = 100 * count / total_pixels
    print(f"- {class_name}: {percentage:.2f}% ({count} pixels)")
```

### Advanced Analysis with CitySeg's Analyzer

```python
import cityseg as cs
from cityseg.analysis import SegmentationAnalyzer
import xarray as xr
import numpy as np

# Create dataset with segmentation result
seg_map = ...  # Your segmentation result
id2label = ...  # Class mapping dictionary

# Create xarray DataArray
seg_data = xr.DataArray(
    np.expand_dims(seg_map, axis=0),  # Add time dimension for consistency
    dims=["time", "y", "x"],
    coords={
        "time": [0],  # Single frame
        "y": np.arange(seg_map.shape[0]),
        "x": np.arange(seg_map.shape[1]),
    },
)

# Create dataset
dataset = xr.Dataset({"segmentation": seg_data})
dataset.attrs = {"id2label": id2label}

# Analyze segmentation
analysis_path = SegmentationAnalyzer.analyze_segmentation_dataset(
    dataset, "results/analysis"
)

# Load and examine analysis results
import pandas as pd
analysis_df = pd.read_parquet(analysis_path)

# Show most common categories
top_categories = analysis_df.sort_values("percentage", ascending=False).head(5)
print("Top 5 categories:")
for _, row in top_categories.iterrows():
    category_id = row["category_id"]
    category_name = id2label.get(category_id, f"Unknown-{category_id}")
    print(f"- {category_name}: {row['percentage']:.2f}%")
```

## Visualization Options

CitySeg provides several visualization options for segmentation results.

### Basic Visualizations

```python
from cityseg.analysis import VisualizationHandler
import numpy as np

# Input image and segmentation map
image = ...  # Original image as numpy array or PIL Image
seg_map = ...  # Segmentation map (class IDs) as numpy array
palette = ...  # Color palette dictionary or array

# 1. Colored segmentation (just the segmentation, no original image)
colored_segmentation = VisualizationHandler.visualize_segmentation(
    image, seg_map, palette, colored_only=True
)

# 2. Transparent overlay on original image
overlay = VisualizationHandler.visualize_segmentation(
    image, seg_map, palette, colored_only=False, alpha=0.6
)

# 3. Contour visualization (just segment boundaries)
contours = VisualizationHandler.visualize_segmentation(
    image, seg_map, palette, colored_only=False, alpha=0.0, contour_width=2
)
```

### Custom Visualization Options

```python
from cityseg.analysis import VisualizationHandler
import numpy as np

# Create a custom palette that highlights specific classes
id2label = ...  # Your class mapping dictionary
original_palette = ...  # Original color palette

# Create a custom palette that highlights roads and sidewalks
custom_palette = original_palette.copy()
road_id = next(k for k, v in id2label.items() if v == "road")
sidewalk_id = next(k for k, v in id2label.items() if v == "sidewalk")

# Make everything except roads and sidewalks grayscale
for class_id in custom_palette.keys():
    if class_id not in [road_id, sidewalk_id]:
        # Convert to grayscale (average of RGB)
        color = custom_palette[class_id]
        gray = int(np.mean(color))
        custom_palette[class_id] = (gray, gray, gray)

# Create visualization with custom palette
custom_viz = VisualizationHandler.visualize_segmentation(
    image, seg_map, custom_palette, colored_only=False, alpha=0.7
)
```

## Tips for Image Processing

### Managing Resolution and Memory

High-resolution images may require significant memory for processing. Use the `max_size` parameter to control resource usage:

```python
# For memory-efficient processing
config = cs.Config(
    input="path/to/high_res_image.jpg",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        max_size=1280  # Resize to a maximum dimension of 1280 pixels
    )
)
```

### Batch Processing for Efficiency

When processing multiple images, create the segmentation pipeline once and reuse it:

```python
# Create pipeline once
segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)

# Process multiple images with the same pipeline
for image_path in image_paths:
    # Process image using existing pipeline
    result = SegmentationProcessor.process_image(
        ImageProcessor.load_image(image_path), 
        segmentation_pipeline
    )
```

### Choosing the Right Model

Different models offer different trade-offs between accuracy and speed:

- **SegFormer-B0**: Fastest, good for real-time applications or large batches
- **SegFormer-B5**: More accurate but slower
- **OneFormer**: Best quality but slower processing
- **Mask2Former**: Good balance of quality and speed

```python
# Fast processing with smaller model
fast_config = cs.Config(
    input="path/to/image.jpg",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        max_size=640  # Lower resolution for faster processing
    )
)

# High quality with larger model
high_quality_config = cs.Config(
    input="path/to/image.jpg",
    model=cs.ModelConfig(
        name="shi-labs/oneformer_cityscapes_swin_large",
        max_size=1920  # Higher resolution for better quality
    )
)
```

# Video Processing Guide

This guide explains how to use CitySeg for processing video files, extracting frames, generating segmentation maps, and analyzing the results.

## Basic Video Processing

Processing a single video with CitySeg can be done using either the legacy interface or the component-based architecture.

### Using the Legacy Interface

The simplest way to process a video is with the legacy interface:

```python
import cityseg as cs

# Create or load configuration for video processing
config = cs.Config(
    input="path/to/your/video.mp4",        # Input video file
    output_dir="results",                  # Output directory
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ),
    frame_step=5,                         # Process every 5th frame
    save_overlay=True,                    # Create overlay visualization
    save_colored_segmentation=True        # Save colored segmentation
)

# Create processor and process the video
cs.create_processor(config).process()
```

The process will:
1. Extract frames at the specified interval
2. Process each frame through the segmentation model
3. Save the results according to your configuration

### Using the Component Architecture

For more control over the video processing pipeline, use the component-based architecture:

```python
import cityseg as cs
from cityseg.components import VideoProcessor, ImageProcessor, SegmentationProcessor
from cityseg.analysis import VisualizationHandler
from cityseg.storage import StorageAdapter
import numpy as np
import xarray as xr
from pathlib import Path

# Create configuration
config = cs.Config.from_yaml("config.yaml")

# Get video metadata
metadata = VideoProcessor.get_metadata(config.input)
print(f"Video metadata: {metadata}")

# Calculate frame indices to process (every Nth frame)
frame_step = config.frame_step
frame_indices = list(range(0, metadata["frame_count"], frame_step))

# Create segmentation pipeline
segmentation_pipeline = SegmentationProcessor.create_pipeline(config.model)

# Process frames in batches
batch_size = 10  # Process 10 frames at a time
segmentation_results = []
frame_batches = [frame_indices[i:i+batch_size] for i in range(0, len(frame_indices), batch_size)]

for batch_indices in frame_batches:
    # Extract batch of frames
    frames = VideoProcessor.get_frames(config.input, batch_indices)
    
    # Process each frame
    for i, frame in enumerate(frames):
        # Resize frame for processing
        resized_frame = ImageProcessor.resize_image(frame, config.model.max_size)
        
        # Process frame
        result = SegmentationProcessor.process_image(resized_frame, segmentation_pipeline)
        
        # Store results
        result["frame_index"] = batch_indices[i]
        segmentation_results.append(result)

# Organize results into dataset
seg_maps = np.stack([r["seg_map"] for r in segmentation_results])
frame_indices = [r["frame_index"] for r in segmentation_results]
id2label = segmentation_results[0].get("id2label", {})

# Create xarray dataset for analysis and storage
seg_data = xr.DataArray(
    seg_maps,
    dims=["time", "y", "x"],
    coords={
        "time": frame_indices,
        "y": np.arange(seg_maps.shape[1]),
        "x": np.arange(seg_maps.shape[2]),
    },
)

# Create dataset
dataset = xr.Dataset({"segmentation": seg_data})
dataset.attrs = {
    "video_path": str(config.input),
    "frame_count": metadata["frame_count"],
    "fps": metadata["fps"],
    "id2label": id2label,
    "frame_step": frame_step,
}

# Save results
storage = StorageAdapter()
output_path = Path(config.output_dir) / f"{Path(config.input).stem}_segmentation"
zarr_path = storage.save_segmentation_data(dataset, dataset.attrs, output_path)
print(f"Saved segmentation data to {zarr_path}")

# Generate visualizations if requested
if config.save_overlay or config.save_colored_segmentation:
    # Process original frames for visualization
    for i, result in enumerate(segmentation_results):
        frame_index = result["frame_index"]
        seg_map = result["seg_map"]
        palette = result.get("palette")
        
        # Get original frame for overlay
        original_frame = VideoProcessor.get_frames(config.input, [frame_index])[0]
        
        if config.save_overlay:
            # Create overlay
            overlay = VisualizationHandler.visualize_segmentation(
                original_frame, seg_map, palette, 
                colored_only=False, alpha=config.visualization.alpha
            )
            
            # Save overlay frame
            overlay_path = Path(config.output_dir) / "overlay_frames" / f"frame_{frame_index:06d}.png"
            overlay_path.parent.mkdir(exist_ok=True, parents=True)
            ImageProcessor.save_image(overlay, overlay_path)
            
        if config.save_colored_segmentation:
            # Create colored segmentation
            colored_seg = VisualizationHandler.visualize_segmentation(
                original_frame, seg_map, palette, colored_only=True
            )
            
            # Save colored segmentation frame
            colored_path = Path(config.output_dir) / "colored_frames" / f"frame_{frame_index:06d}.png"
            colored_path.parent.mkdir(exist_ok=True, parents=True)
            ImageProcessor.save_image(colored_seg, colored_path)
    
    print(f"Saved visualization frames to {config.output_dir}")
```

## Processing Multiple Videos

CitySeg can efficiently process multiple videos by pointing to a directory containing video files.

### Using the Legacy Interface

```python
import cityseg as cs

# Configure for a directory of videos
config = cs.Config(
    input="path/to/video_directory",     # Directory containing videos
    output_dir="results",               # Output directory
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ),
    frame_step=10                      # Process every 10th frame
)

# Process all videos in the directory
cs.create_processor(config).process()
```

### Processing Videos with Custom Logic

```python
import cityseg as cs
from cityseg.components import SegmentationProcessor
from pathlib import Path

# Create model configuration
model_config = cs.ModelConfig(
    name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    device="auto"
)

# Create segmentation pipeline once for all videos
segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)

# Get all video files
video_directory = Path("path/to/videos")
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
video_paths = []

for ext in video_extensions:
    video_paths.extend(video_directory.glob(f"*{ext}"))

# Process each video
for video_path in video_paths:
    # Create configuration for this video
    config = cs.Config(
        input=str(video_path),
        output_dir=f"results/{video_path.stem}",
        model=model_config,
        frame_step=10
    )
    
    # Create processor with existing pipeline
    processor = cs.create_processor(config)
    processor.process()
    
    print(f"Processed {video_path.name}")
```

## Analyzing Video Segmentation Results

After processing a video, you can analyze the segmentation results to extract temporal patterns and insights.

### Basic Temporal Analysis

```python
import cityseg as cs
from cityseg.storage import StorageAdapter
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

# Load segmentation results from Zarr storage
storage = StorageAdapter()
segmentation_data = storage.load_segmentation_data("results/video_segmentation.zarr")

# Get id2label mapping
id2label = segmentation_data.attrs.get("id2label", {})

# Analyze class distribution over time
seg_array = segmentation_data.segmentation.values

# Create a list to store per-frame analysis
frame_analyses = []

# Analyze each frame
for i, frame_idx in enumerate(segmentation_data.time.values):
    frame_seg = seg_array[i]
    
    # Count pixels per class
    unique_classes, counts = np.unique(frame_seg, return_counts=True)
    total_pixels = frame_seg.size
    
    # Create a dictionary for this frame
    frame_data = {
        "frame": int(frame_idx),
        "total_pixels": total_pixels
    }
    
    # Add percentage for each class
    for cls_id, count in zip(unique_classes, counts):
        class_name = id2label.get(int(cls_id), f"Unknown-{cls_id}")
        percentage = 100 * count / total_pixels
        frame_data[class_name] = percentage
    
    frame_analyses.append(frame_data)

# Create DataFrame from analyses
analysis_df = pd.DataFrame(frame_analyses)

# Plot temporal trends for top classes
top_classes = [
    col for col in analysis_df.columns 
    if col not in ["frame", "total_pixels"]
]

top_classes = sorted(
    top_classes, 
    key=lambda x: analysis_df[x].mean(), 
    reverse=True
)[:5]  # Top 5 classes by average presence

# Plot the trends
plt.figure(figsize=(12, 6))

for cls in top_classes:
    plt.plot(analysis_df["frame"], analysis_df[cls], label=cls)

plt.xlabel("Frame Number")
plt.ylabel("Class Coverage (%)")
plt.title("Semantic Class Distribution Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/temporal_analysis.png")
plt.close()

print(f"Created temporal analysis plot: results/temporal_analysis.png")
```

### Advanced Video Analysis with CitySeg's Analyzer

```python
import cityseg as cs
from cityseg.analysis import SegmentationAnalyzer
from cityseg.storage import StorageAdapter
import pandas as pd
import matplotlib.pyplot as plt

# Load segmentation data
storage = StorageAdapter()
dataset = storage.load_segmentation_data("results/video_segmentation.zarr")

# Run comprehensive analysis
analysis_path = SegmentationAnalyzer.analyze_segmentation_dataset(
    dataset, "results/video_analysis"
)

# Load analysis results
analysis_df = pd.read_parquet(analysis_path)

# Get category mapping
id2label = dataset.attrs.get("id2label", {})

# Create summary statistics
category_summary = analysis_df.groupby("category_id").agg({
    "percentage": ["mean", "std", "min", "max"],
    "pixel_count": ["mean", "sum"]
}).reset_index()

# Add category names
category_summary["category_name"] = category_summary["category_id"].map(
    lambda x: id2label.get(x, f"Unknown-{x}")
)

# Print summary
print("Category Summary Statistics:")
print(category_summary[["category_name", ("percentage", "mean"), ("percentage", "std")]])

# Plot temporal trends for top 5 categories
top_categories = [
    int(x) for x in category_summary.sort_values(
        [("percentage", "mean")], ascending=False
    )["category_id"].head(5)
]

plt.figure(figsize=(12, 6))

for cat_id in top_categories:
    cat_data = analysis_df[analysis_df["category_id"] == cat_id]
    cat_name = id2label.get(cat_id, f"Unknown-{cat_id}")
    plt.plot(cat_data["time"], cat_data["percentage"], label=cat_name)

plt.xlabel("Frame")
plt.ylabel("Coverage (%)")
plt.title("Top 5 Categories: Temporal Trends")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/video_category_trends.png")
plt.close()

# Create stacked area chart for composition
top_cat_data = analysis_df[analysis_df["category_id"].isin(top_categories)].copy()
top_cat_data["category_name"] = top_cat_data["category_id"].map(
    lambda x: id2label.get(x, f"Unknown-{x}")
)

# Pivot data for stacked area chart
pivot_data = top_cat_data.pivot(index="time", columns="category_name", values="percentage")

# Plot stacked area chart
plt.figure(figsize=(12, 6))
plt.stackplot(
    pivot_data.index, 
    [pivot_data[col] for col in pivot_data.columns],
    labels=pivot_data.columns,
    alpha=0.8
)

plt.xlabel("Frame")
plt.ylabel("Composition (%)")
plt.title("Video Scene Composition Over Time")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.savefig("results/video_composition.png")
plt.close()

print("Analysis visualizations created.")
```

## Working with Video Frames

CitySeg provides several ways to work with video frames directly.

### Extracting Specific Frames

```python
from cityseg.components import VideoProcessor
import matplotlib.pyplot as plt

# Extract specific frames by index
video_path = "path/to/video.mp4"
frame_indices = [0, 100, 200, 300]  # Get specific frames

# Extract frames
frames = VideoProcessor.get_frames(video_path, frame_indices)

# Display frames
plt.figure(figsize=(15, 4))

for i, frame in enumerate(frames):
    plt.subplot(1, len(frames), i+1)
    plt.imshow(frame)
    plt.title(f"Frame {frame_indices[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("extracted_frames.png")
plt.close()

print(f"Extracted {len(frames)} frames")
```

### Processing Frames in Batches

```python
from cityseg.components import VideoProcessor, ImageProcessor, SegmentationProcessor
from cityseg.analysis import VisualizationHandler
from pathlib import Path
import tqdm

# Create configuration
video_path = "path/to/video.mp4"
model_config = cs.ModelConfig(
    name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
)
output_dir = Path("results/frame_batches")
output_dir.mkdir(exist_ok=True, parents=True)

# Get video metadata
metadata = VideoProcessor.get_metadata(video_path)
frame_step = 10  # Process every 10th frame
frame_count = metadata["frame_count"]
frame_indices = list(range(0, frame_count, frame_step))

# Create segmentation pipeline
segmentation_pipeline = SegmentationProcessor.create_pipeline(model_config)

# Process in batches of 20 frames
batch_size = 20
frame_batches = [frame_indices[i:i+batch_size] for i in range(0, len(frame_indices), batch_size)]

for batch_idx, batch_indices in enumerate(tqdm.tqdm(frame_batches)):
    # Get batch of frames
    batch_frames = VideoProcessor.get_frames(video_path, batch_indices)
    
    # Process each frame in the batch
    for i, (frame, frame_idx) in enumerate(zip(batch_frames, batch_indices)):
        # Resize for processing
        resized_frame = ImageProcessor.resize_image(frame, model_config.max_size)
        
        # Process frame
        result = SegmentationProcessor.process_image(resized_frame, segmentation_pipeline)
        
        # Create visualization
        overlay = VisualizationHandler.visualize_segmentation(
            frame, result["seg_map"], result.get("palette"), alpha=0.6
        )
        
        # Save result
        output_path = output_dir / f"frame_{frame_idx:06d}.png"
        ImageProcessor.save_image(overlay, output_path)
    
    print(f"Processed batch {batch_idx+1}/{len(frame_batches)}")
```

### Creating Output Videos from Processed Frames

```python
from cityseg.components import VideoProcessor
import cv2
from pathlib import Path
import numpy as np

# Input video path (for metadata)
video_path = "path/to/video.mp4"

# Get processed frame paths (assuming they've been saved with frame numbers in filenames)
frames_dir = Path("results/frame_batches")
frame_paths = sorted(list(frames_dir.glob("frame_*.png")))

# Get original video metadata
metadata = VideoProcessor.get_metadata(video_path)
original_fps = metadata["fps"]
frame_step = 10  # The step used during processing

# Create output video writer
output_video_path = "results/segmentation_video.mp4"

# Get first frame to determine dimensions
first_frame = cv2.imread(str(frame_paths[0]))
height, width = first_frame.shape[:2]

# Create video writer
output_fps = original_fps / frame_step  # Adjust FPS based on frame step
video_writer = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    output_fps,
    (width, height)
)

# Add frames to video
for frame_path in frame_paths:
    frame = cv2.imread(str(frame_path))
    video_writer.write(frame)

# Release the video writer
video_writer.release()

print(f"Created output video: {output_video_path}")
```

## Tips for Video Processing

### Memory Management for Long Videos

When processing long videos, process frames in smaller batches to manage memory efficiently:

```python
# Efficient batch processing for long videos
small_batch_size = 5  # Process fewer frames at once
many_batches = [frame_indices[i:i+small_batch_size] for i in range(0, len(frame_indices), small_batch_size)]

for batch in many_batches:
    # Process small batch of frames
    frames = VideoProcessor.get_frames(video_path, batch)
    # Process frames...
    
    # Clear memory after each batch
    import gc
    gc.collect()
```

### Balancing Frame Rate and Processing Time

Selecting an appropriate `frame_step` can balance processing time with temporal resolution:

```python
# Fast processing with less temporal detail
fast_config = cs.Config(
    input="path/to/video.mp4",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        max_size=640  # Lower resolution
    ),
    frame_step=30  # Process 1 frame per second at 30 fps
)

# Detailed processing (more temporal resolution)
detailed_config = cs.Config(
    input="path/to/video.mp4",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        max_size=1280
    ),
    frame_step=5  # Process every 5th frame
)
```

### Choosing the Right Model for Video

For video processing, consider model speed and accuracy trade-offs:

```python
# Fast, lightweight model for long videos
fast_model_config = cs.ModelConfig(
    name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",  # Smaller model
    max_size=640,  # Lower resolution
    device="auto"
)

# High-quality model for shorter videos
high_quality_model_config = cs.ModelConfig(
    name="shi-labs/oneformer_cityscapes_swin_large",  # Larger, more accurate model
    max_size=1280,  # Higher resolution
    device="auto"
)
```

### Processing 4K Videos Efficiently

For high-resolution videos, consider aggressive downsampling during processing:

```python
# Efficient processing of 4K videos
config = cs.Config(
    input="path/to/4k_video.mp4",
    model=cs.ModelConfig(
        name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        max_size=960  # Downsample significantly for memory efficiency
    ),
    frame_step=20,  # Process fewer frames
    batch_size=4  # Small batch size for memory management
)
```

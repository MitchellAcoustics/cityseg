"""Demo of the new media-first CitySeg architecture.

This script demonstrates the media-first workflow:
1. Load image/video as MediaDataset
2. Apply segmentation analysis to existing dataset
3. Use all the enhanced functionality together
4. Compare with the traditional workflow
"""
# %%

import tempfile
from pathlib import Path

import cityseg

# %%
print("=== New Media-First Workflow Demo ===")

test_image = (
    "/Users/mitch/Documents/GitHub/cityseg/example_inputs/EustonTap-Screenshot1.png"
)

print("\n1. Load image as MediaDataset (no segmentation yet)")
media_ds = cityseg.load_image(test_image)
media_ds

# %%
print("\n2. Apply segmentation to existing MediaDataset")
model, processor = cityseg.load_segmentation_model()

# Option 1: Using the standalone function
segmented_ds = cityseg.apply_segmentation(
    media_ds=media_ds, model=model, processor=processor, return_confidence=True
)

# Option 2: Using the new accessor method (same result!)
segmented_ds_v2 = media_ds.seg.apply_segmentation(
    model=model, processor=processor, return_confidence=True
)

print(f"Standalone function result: {segmented_ds.sizes}")
print(f"Accessor method result: {segmented_ds_v2.sizes}")
print(f"Results are identical: {list(segmented_ds.data_vars) == list(segmented_ds_v2.data_vars)}")

segmented_ds

# %%
print("\n3. Use segmentation accessor with media-first dataset")
print(f"Number of classes: {segmented_ds.seg.num_classes}")

# Show class distribution
stats = segmented_ds.seg.class_statistics()
print("\nTop 5 classes by area:")
sorted_stats = sorted(stats.items(), key=lambda x: x[1]["percentage"], reverse=True)[:5]
for class_name, class_info in sorted_stats:
    print(f"  {class_name}: {class_info['percentage']:.1f}%")

# %%
print("\n4. Demonstrate different visualization types")

# All plots now work because image data is guaranteed to be present
print("Segmentation overlay on original image:")
segmented_ds.seg.plot(kind="overlay", alpha=0.7)

print("Side-by-side comparison:")
segmented_ds.seg.plot(kind="side-by-side")

print("Class distribution:")
segmented_ds.seg.plot(kind="stats")

# %%
print("\n5. Pipeline shortcuts for convenience")

print("Using segment_image_file() pipeline:")
quick_result = cityseg.segment_image_file(
    test_image, model=model, processor=processor, return_confidence=True
)
print(f"Quick result: {quick_result.sizes}")
print(
    f"Same result as media-first approach: {list(quick_result.data_vars) == list(segmented_ds.data_vars)}"
)

# %%
print("\n6. Working with media without segmentation")

print("MediaDataset can be used for image analysis without segmentation:")
print(f"Image shape: {media_ds.image.shape}")
print(f"Image data type: {media_ds.image.dtype}")
print(f"RGB channels: {list(media_ds.rgb.values)}")

# Can plot just the image before applying segmentation
print("Plotting image from MediaDataset:")
media_ds.seg.plot(kind="image")

# Can save/load MediaDatasets independently
with tempfile.TemporaryDirectory() as tmp_dir:
    media_path = Path(tmp_dir) / "media.zarr"
    cityseg.save_media_dataset(media_ds, media_path)

    loaded_media = cityseg.load_media_dataset(media_path)
    print(f"Saved and loaded MediaDataset: {loaded_media.sizes}")

# %%
print("\n8. Video example (if you have a video file)")
# This would work the same way for videos:
# video_ds = cityseg.load_video("path/to/video.mp4", max_frames=30)
# segmented_video = cityseg.apply_segmentation(video_ds, model, processor)

print("Video workflow:")
print("1. video_ds = cityseg.load_video('video.mp4')")
print("2. segmented_video = video_ds.seg.apply_segmentation(model, processor)")
print("3. segmented_video.seg.plot(kind='stats')")
print("\nOr using standalone function:")
print("2. segmented_video = cityseg.apply_segmentation(video_ds, model, processor)")

print("\n🎉 Media-first architecture demo complete!")

"""xarray Accessor for SegmentationDatasets.

This module provides the .seg accessor for xarray Datasets, enabling
specialized operations on segmentation data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from cityseg.components.media import MediaDataset

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


@xr.register_dataset_accessor("seg")
class SegmentationAccessor:
    """xarray accessor for segmentation-specific operations.

    Provides convenient methods for analyzing and visualizing segmentation data.
    Access via: dataset.seg.method_name()

    Example:
        ds = load_segmentation_dataset("path/to/data.zarr")

        # Get class statistics
        stats = ds.seg.class_stats()

        # Create visualization
        ds.seg.plot_segmentation(frame=0)

        # Extract region of interest
        roi = ds.seg.crop(x_slice=slice(100, 400), y_slice=slice(50, 300))
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

        # Validate that this is a SegmentationDataset
        if not isinstance(xarray_obj, xr.Dataset):
            raise TypeError(
                "SegmentationAccessor can only be used with xarray.Dataset objects"
            )

        MediaDataset.validate_dataset(xarray_obj)

    @property
    def is_video(self) -> bool:
        """Check if this is video data (has time dimension)."""
        return "time" in self._obj.dims

    @property
    def has_segmentation(self) -> bool:
        """Check if this dataset has segmentation data."""
        return "seg_map" in self._obj.data_vars

    @property
    def is_media_dataset(self) -> bool:
        """Check if this is a MediaDataset (has image data)."""
        return "image" in self._obj.data_vars

    @property
    def num_classes(self) -> int:
        """Get the number of unique classes in the segmentation."""
        if not self.has_segmentation:
            raise ValueError(
                "This property requires segmentation data. Use apply_segmentation() first."
            )

        if hasattr(self._obj, "class_labels"):
            return len(self._obj.attrs["class_labels"])
        else:
            # Compute from data
            unique_labels = np.unique(self._obj.seg_map.values)
            return len(unique_labels)

    def get_class_label_map(self) -> xr.DataArray:
        """Convert numerical classes in seg_map to their text labels"""
        if not self.has_segmentation:
            raise ValueError(
                "This method requires segmentation data. Use apply_segmentation() first."
            )
        return self._obj.class_label.sel(class_id=self._obj.seg_map)

    def get_colored_segmentation(self) -> xr.DataArray:
        """Convert seg_map to RGB visualization using the palette"""
        if not self.has_segmentation:
            raise ValueError(
                "This method requires segmentation data. Use apply_segmentation() first."
            )
        return self._obj.palette.sel(class_id=self._obj.seg_map)

    def class_statistics(self) -> dict[str, dict[str, float]]:
        """Calculate statistics for each class in the segmentation map"""
        if not self.has_segmentation:
            raise ValueError(
                "This method requires segmentation data. Use apply_segmentation() first."
            )
        seg_map = self._obj.seg_map.values
        total_pixels = seg_map.size

        unique_classes, counts = np.unique(seg_map, return_counts=True)

        stats = {}
        for cls_id, count in zip(unique_classes, counts):
            # Get class label if available
            if "class_label" in self._obj.data_vars and cls_id < len(
                self._obj.class_label
            ):
                class_label = self._obj.class_label.sel(class_id=cls_id).item()
            else:
                class_label = f"Unknown-{cls_id}"

            # Calculate statistics
            percentage = 100 * count / total_pixels

            stats[class_label] = {
                "class_id": int(cls_id),
                "pixel_count": int(count),
                "percentage": float(percentage),
            }

        return stats

    def plot(
        self,
        kind: str = "image",
        alpha: float = 0.6,
        figsize: tuple[float, float] = (12, 8),
    ) -> None:
        """Plot dataset contents (image, segmentation, or both)

        Parameters
        ----------
        kind : str
            Type of plot: 'image', 'overlay', 'segmentation', 'side-by-side', or 'stats'
            - 'image': Show just the original image (works for MediaDataset)
            - 'overlay': Overlay segmentation on image (requires segmentation)
            - 'segmentation': Show just the segmentation (requires segmentation)
            - 'side-by-side': Image and segmentation side by side (requires segmentation)
            - 'stats': Class distribution chart (requires segmentation)
        alpha : float
            Transparency for overlay visualization
        figsize : tuple
            Figure size (width, height) in inches
        """
        plt.figure(figsize=figsize)

        if kind == "image":
            # Show just the original image
            plt.imshow(self._obj.image.values)
            plt.title("Original Image")
            plt.axis("off")

        elif kind == "overlay":
            # Create overlay of segmentation on original image
            if "image" in self._obj.data_vars:
                plt.imshow(self._obj.image.values)
                colored_seg = self.get_colored_segmentation().values
                plt.imshow(colored_seg, alpha=alpha)
                plt.title("Segmentation Overlay")
            else:
                raise ValueError(
                    "Dataset must contain 'image' data variable for overlay plot"
                )
            plt.axis("off")

        elif kind == "segmentation":
            # Show just the colored segmentation
            plt.imshow(self.get_colored_segmentation().values)
            plt.title("Segmentation Map")
            plt.axis("off")

        elif kind == "side-by-side":
            # Show original and segmentation side by side
            if "image" in self._obj.data_vars:
                plt.subplot(1, 2, 1)
                plt.imshow(self._obj.image.values)
                plt.title("Original Image")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(self.get_colored_segmentation().values)
                plt.title("Segmentation Map")
                plt.axis("off")
            else:
                raise ValueError(
                    "Dataset must contain 'image' data variable for side-by-side plot"
                )

        elif kind == "stats":
            # Show class percentages as a bar chart
            stats = self.class_statistics()
            # Sort by percentage
            sorted_stats = sorted(
                stats.items(), key=lambda x: x[1]["percentage"], reverse=True
            )

            # Get top 10 classes
            top_classes = sorted_stats[:10]

            labels = [class_name for class_name, _ in top_classes]
            percentages = [stats_data["percentage"] for _, stats_data in top_classes]
            colors = []

            # Try to get class colors from palette
            for class_name, class_info in top_classes:
                cls_id = class_info["class_id"]
                if "palette" in self._obj.data_vars and cls_id < len(self._obj.palette):
                    # Convert RGB (0-255) to matplotlib format (0-1)
                    rgb = self._obj.palette.sel(class_id=cls_id).values / 255.0
                    colors.append(rgb)
                else:
                    colors.append(None)  # Use default color

            plt.barh(labels, percentages, color=colors)
            plt.xlabel("Area Percentage (%)")
            plt.title("Class Distribution")
            plt.tight_layout()

        else:
            raise ValueError(
                f"Unknown plot kind: {kind}. Available: 'image', 'overlay', 'segmentation', 'side-by-side', 'stats'"
            )

        plt.show()

    @property
    def class_labels(self) -> dict[int, str]:
        """Get the class label mapping."""
        if "class_labels" in self._obj.attrs:
            return self._obj.attrs["class_labels"]
        else:
            # Generate default labels
            unique_labels = np.unique(self._obj.seg_map.values)
            return {int(label): f"class_{label}" for label in unique_labels}

    def class_stats(self, normalize: bool = False) -> pd.DataFrame:
        """Compute statistics for each class across the dataset.

        Args:
            normalize: If True, return proportions instead of counts

        Returns:
            DataFrame with class statistics (counts/proportions, percentages)
        """
        seg_data = self._obj.seg_map

        if self.is_video:
            # For video data, compute stats per frame then aggregate
            frame_stats = []
            for frame_idx in range(seg_data.sizes["time"]):
                frame_data = seg_data.isel(time=frame_idx)
                unique, counts = np.unique(frame_data.values, return_counts=True)

                total_pixels = frame_data.size
                frame_df = pd.DataFrame(
                    {
                        "class_id": unique,
                        "pixel_count": counts,
                        "percentage": (counts / total_pixels) * 100,
                        "frame": frame_idx,
                    }
                )
                frame_stats.append(frame_df)

            # Combine all frames
            all_stats = pd.concat(frame_stats, ignore_index=True)

            # Aggregate statistics
            agg_stats = (
                all_stats.groupby("class_id")
                .agg(
                    {
                        "pixel_count": ["sum", "mean", "std"],
                        "percentage": ["mean", "std", "min", "max"],
                    }
                )
                .round(2)
            )

            # Flatten column names
            agg_stats.columns = ["_".join(col).strip() for col in agg_stats.columns]

        else:
            # For single image
            unique, counts = np.unique(seg_data.values, return_counts=True)
            total_pixels = seg_data.size

            agg_stats = pd.DataFrame(
                {
                    "pixel_count_sum": counts,
                    "percentage_mean": (counts / total_pixels) * 100,
                },
                index=unique,
            )
            agg_stats.index.name = "class_id"

        # Add class labels
        class_labels = self.class_labels
        agg_stats["class_name"] = [
            class_labels.get(int(class_id), f"unknown_{class_id}")
            for class_id in agg_stats.index
        ]

        if normalize:
            # Convert counts to proportions
            count_cols = [col for col in agg_stats.columns if "pixel_count" in col]
            for col in count_cols:
                if col in agg_stats.columns:
                    total = (
                        agg_stats[col].sum() if "sum" in col else agg_stats[col].mean()
                    )
                    agg_stats[col] = agg_stats[col] / total

        return agg_stats.reset_index()

    def plot_segmentation(
        self,
        frame: int | None = None,
        ax: Axes | None = None,
        use_palette: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot segmentation visualization.

        Args:
            frame: Frame index for video data (required for video)
            ax: Matplotlib axes to plot on (creates new if None)
            use_palette: Use color palette from dataset attributes if available
            **kwargs: Additional arguments passed to matplotlib.pyplot.imshow

        Returns:
            Matplotlib axes with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 8)))

        # Get segmentation data
        seg_data = self._obj.seg_map

        if self.is_video:
            if frame is None:
                raise ValueError("frame parameter required for video data")
            if frame >= seg_data.sizes["time"]:
                raise ValueError(
                    f"Frame {frame} >= num_frames {seg_data.sizes['time']}"
                )
            plot_data = seg_data.isel(time=frame)
            title = f"Segmentation - Frame {frame}"
        else:
            if frame is not None:
                logger.warning("frame parameter ignored for single image data")
            plot_data = seg_data
            title = "Segmentation"

        # Create color mapping
        if use_palette and "palette" in self._obj.attrs:
            palette = self._obj.attrs["palette"]
            # Convert palette to colormap
            unique_labels = np.unique(plot_data.values)
            colors = []
            for label in unique_labels:
                if label in palette:
                    colors.append(
                        [c / 255.0 for c in palette[label]]
                    )  # Normalize to 0-1
                else:
                    # Default color for missing labels
                    colors.append([0.5, 0.5, 0.5])

            # Create custom colormap
            from matplotlib.colors import ListedColormap

            cmap = ListedColormap(colors)
            kwargs.setdefault("cmap", cmap)
            kwargs.setdefault("vmin", unique_labels.min())
            kwargs.setdefault("vmax", unique_labels.max())
        else:
            kwargs.setdefault("cmap", "tab20")

        # Plot
        im = ax.imshow(plot_data.values, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        # Add colorbar with class labels if possible
        if "class_labels" in self._obj.attrs:
            cbar = plt.colorbar(im, ax=ax)
            class_labels = self.class_labels

            # Set colorbar ticks to class IDs
            unique_labels = np.unique(plot_data.values)
            cbar.set_ticks(unique_labels)
            cbar.set_ticklabels(
                [class_labels.get(int(label), str(label)) for label in unique_labels]
            )
            cbar.set_label("Class")

        return ax

    def crop(
        self,
        x_slice: slice | None = None,
        y_slice: slice | None = None,
        time_slice: slice | None = None,
    ) -> xr.Dataset:
        """Crop the segmentation dataset to a region of interest.

        Args:
            x_slice: Slice for x dimension (width)
            y_slice: Slice for y dimension (height)
            time_slice: Slice for time dimension (frames, video only)

        Returns:
            Cropped Dataset
        """
        slices = {}
        if x_slice is not None:
            slices["x"] = x_slice
        if y_slice is not None:
            slices["y"] = y_slice
        if time_slice is not None and self.is_video:
            slices["time"] = time_slice

        if not slices:
            logger.warning("No slices provided, returning original dataset")
            return self._obj

        cropped = self._obj.isel(**slices)
        logger.info(f"Cropped dataset from {self._obj.sizes} to {cropped.sizes}")
        return cropped

    def mask_by_class(self, class_ids: int | list[int]) -> xr.Dataset:
        """Create a binary mask for specific classes.

        Args:
            class_ids: Single class ID or list of class IDs to include in mask

        Returns:
            Dataset with binary mask (1 where class matches, 0 elsewhere)
        """
        if isinstance(class_ids, int):
            class_ids = [class_ids]

        seg_data = self._obj.seg_map
        mask = xr.zeros_like(seg_data, dtype=bool)

        for class_id in class_ids:
            mask = mask | (seg_data == class_id)

        # Create new dataset with mask
        mask_ds = self._obj.copy()
        mask_ds["mask"] = mask.astype(int)

        # Update attributes
        class_labels = self.class_labels
        selected_labels = {
            cid: class_labels.get(cid, f"class_{cid}") for cid in class_ids
        }
        mask_ds.attrs["mask_classes"] = selected_labels

        logger.info(f"Created mask for classes: {selected_labels}")
        return mask_ds

    def temporal_stats(self) -> xr.Dataset:
        """Compute temporal statistics for video data.

        Returns:
            Dataset with temporal statistics (mean, std, etc. over time)

        Raises:
            ValueError: If dataset is not video data
        """
        if not self.is_video:
            raise ValueError("Temporal statistics only available for video data")

        seg_data = self._obj.seg_map

        # Compute statistics over time dimension
        # Most frequent class per pixel (mode calculation)
        def compute_mode(arr):
            """Compute mode along first axis (time)."""
            from scipy import stats

            # arr shape is (time, y, x) -> we want mode along axis 0
            mode_result = stats.mode(arr, axis=0, keepdims=False)
            return mode_result.mode

        class_mode = xr.apply_ufunc(
            compute_mode,
            seg_data,
            input_core_dims=[["time"]],
            output_core_dims=[[]],
            dask="forbidden",  # Disable dask for now to avoid complications
        )

        stats_ds = xr.Dataset(
            {
                "class_mode": class_mode,  # Most frequent class per pixel
                "class_variability": seg_data.std(dim="time"),  # Variability over time
                "num_class_changes": (seg_data.diff(dim="time") != 0).sum(
                    dim="time"
                ),  # Number of class changes
            }
        )

        # Copy coordinates and attributes
        stats_ds.coords.update(
            {k: v for k, v in self._obj.coords.items() if k != "time"}
        )
        stats_ds.attrs.update(self._obj.attrs)
        stats_ds.attrs["temporal_analysis"] = True

        logger.info("Computed temporal statistics")
        return stats_ds

    def to_rgb(
        self,
        frame: int | None = None,
        palette: dict[int, tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """Convert segmentation to RGB image using color palette.

        Args:
            frame: Frame index for video data
            palette: Color palette mapping {class_id: (r, g, b)}. Uses dataset
                    palette if None provided.

        Returns:
            RGB image array with shape (H, W, 3)
        """
        # Get segmentation data
        seg_data = self._obj.seg_map

        if self.is_video:
            if frame is None:
                raise ValueError("frame parameter required for video data")
            seg_array = seg_data.isel(time=frame).values
        else:
            seg_array = seg_data.values

        # Get color palette
        if palette is None:
            if "palette" in self._obj.attrs:
                palette = self._obj.attrs["palette"]
            else:
                # Generate default palette
                unique_labels = np.unique(seg_array)
                import matplotlib.pyplot as plt

                cmap = plt.cm.get_cmap("tab20")
                palette = {}
                for i, label in enumerate(unique_labels):
                    color = cmap(i / len(unique_labels))
                    r, g, b = color[:3]
                    palette[int(label)] = (int(r * 255), int(g * 255), int(b * 255))

        # Create RGB image
        h, w = seg_array.shape
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

        if palette is not None:
            for class_id, color in palette.items():
                mask = seg_array == class_id
                rgb_image[mask] = color

        return rgb_image

    def export_analysis(self, output_path: str | Path) -> None:
        """Export comprehensive analysis to files.

        Args:
            output_path: Base path for output files (without extension)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export class statistics
        stats = self.class_stats()
        stats.to_csv(f"{output_path}_class_stats.csv", index=False)

        # Export temporal statistics if video
        if self.is_video:
            temporal_stats = self.temporal_stats()
            temporal_stats.to_netcdf(f"{output_path}_temporal_stats.nc")

        # Export metadata
        metadata = {
            "dataset_info": {
                "dimensions": dict(self._obj.sizes),
                "is_video": self.is_video,
                "num_classes": self.num_classes,
            },
            "attributes": self._obj.attrs,
        }

        import json

        with open(f"{output_path}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Exported analysis to {output_path}_*")

    def to_binary_mask(self, class_id: int | list[int]) -> np.ndarray:
        """Create a binary mask for the specified class ID(s)

        Parameters
        ----------
        class_id : int or list of int
            Class ID(s) to include in the binary mask

        Returns
        -------
        np.ndarray
            Binary mask where 1 indicates the specified class(es)
        """
        seg_map = self._obj.seg_map.values

        if isinstance(class_id, int):
            return (seg_map == class_id).astype(np.uint8)
        elif isinstance(class_id, (list, tuple)):
            mask = np.zeros_like(seg_map, dtype=np.uint8)
            for cid in class_id:
                mask = np.logical_or(mask, seg_map == cid)
            return mask.astype(np.uint8)
        else:
            raise TypeError("class_id must be an integer or list of integers")

    def get_class_boundaries(
        self, class_id: int | list[int] | None = None
    ) -> np.ndarray:
        """Get the boundaries of segmentation regions

        Parameters
        ----------
        class_id : int or list of int, optional
            If provided, only get boundaries for these classes

        Returns
        -------
        np.ndarray
            Binary boundary mask
        """
        from scipy import ndimage

        if class_id is not None:
            # Get binary mask for specified class(es)
            mask = self.to_binary_mask(class_id)
        else:
            # Use full segmentation map
            mask = self._obj.seg_map.values

        # Apply gradient filter to detect edges
        edges_x = ndimage.sobel(mask, axis=0)
        edges_y = ndimage.sobel(mask, axis=1)
        edges = np.hypot(edges_x, edges_y)

        # Normalize and threshold
        edges = (edges > 0).astype(np.uint8)

        return edges

    def apply_segmentation(
        self,
        model=None,
        processor=None,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        return_confidence: bool = False,
    ) -> xr.Dataset:
        """Apply segmentation to this dataset.

        This method provides a convenient way to apply segmentation analysis
        to a MediaDataset using the accessor interface.

        Args:
            model: Pre-loaded segmentation model (loads if None)
            processor: Pre-loaded image processor (loads if None)
            model_name: Model to use if model/processor not provided
            return_confidence: Whether to include confidence scores

        Returns:
            Enhanced dataset with segmentation results added

        Raises:
            ValueError: If dataset doesn't contain image data
            ImportError: If required segmentation dependencies are missing

        Example:
            >>> media_ds = cityseg.load_image("image.jpg")
            >>> segmented_ds = media_ds.seg.apply_segmentation(model, processor)
            >>> stats = segmented_ds.seg.class_statistics()
        """
        # Check that this is a media dataset
        if not self.is_media_dataset:
            raise ValueError(
                "apply_segmentation requires a MediaDataset with 'image' data variable"
            )

        # Import here to avoid circular imports
        from ..segmentation import apply_segmentation

        enhanced_ds = apply_segmentation(
            media_ds=self._obj,
            model=model,
            processor=processor,
            model_name=model_name,
            return_confidence=return_confidence,
        )

        self._obj = enhanced_ds
        return self._obj

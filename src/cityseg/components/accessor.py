"""xarray Accessor for SegmentationDatasets.

This module provides the .seg accessor for xarray Datasets, enabling
specialized operations on segmentation data.
"""

from __future__ import annotations


import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path

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

        # Validate this is a segmentation dataset
        if "segmentation" not in xarray_obj.data_vars:
            raise ValueError("Dataset must contain 'segmentation' data variable")

    @property
    def is_video(self) -> bool:
        """Check if this is video segmentation data (has time dimension)."""
        return "time" in self._obj.dims

    @property
    def num_classes(self) -> int:
        """Get the number of unique classes in the segmentation."""
        if hasattr(self._obj, "class_labels"):
            return len(self._obj.attrs["class_labels"])
        else:
            # Compute from data
            unique_labels = np.unique(self._obj.segmentation.values)
            return len(unique_labels)

    @property
    def class_labels(self) -> dict[int, str]:
        """Get the class label mapping."""
        if "class_labels" in self._obj.attrs:
            return self._obj.attrs["class_labels"]
        else:
            # Generate default labels
            unique_labels = np.unique(self._obj.segmentation.values)
            return {int(label): f"class_{label}" for label in unique_labels}

    def class_stats(self, normalize: bool = False) -> pd.DataFrame:
        """Compute statistics for each class across the dataset.

        Args:
            normalize: If True, return proportions instead of counts

        Returns:
            DataFrame with class statistics (counts/proportions, percentages)
        """
        seg_data = self._obj.segmentation

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
        seg_data = self._obj.segmentation

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

        seg_data = self._obj.segmentation
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

        seg_data = self._obj.segmentation

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
        seg_data = self._obj.segmentation

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

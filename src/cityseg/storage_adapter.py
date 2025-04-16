"""
This module provides adapters for storing and retrieving segmentation data.

It includes classes for handling different storage formats such as Zarr and Parquet,
enabling efficient storage and retrieval of segmentation data and analysis results.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union, Optional

import numpy as np
import pandas as pd
import zarr
from loguru import logger
import xarray as xr

from .config import Config


class SegmentationStorage:
    """
    Abstract base class for segmentation data storage.
    
    This class defines the interface for segmentation storage adapters.
    """
    
    def save_segmentation_data(self, data, metadata: Dict[str, Any], output_path: Path) -> Path:
        """
        Save segmentation data and metadata to storage.
        
        Args:
            data: Segmentation data.
            metadata (Dict[str, Any]): Metadata about the segmentation.
            output_path (Path): Path to save the data.
            
        Returns:
            Path: Path to the saved data.
        """
        raise NotImplementedError("Subclasses must implement save_segmentation_data")
    
    def load_segmentation_data(self, input_path: Path) -> Tuple[Any, Dict[str, Any]]:
        """
        Load segmentation data and metadata from storage.
        
        Args:
            input_path (Path): Path to the saved data.
            
        Returns:
            Tuple[Any, Dict[str, Any]]: Tuple of segmentation data and metadata.
        """
        raise NotImplementedError("Subclasses must implement load_segmentation_data")
    
    def load_segmentation_batch(self, input_path: Path, start: int, end: int) -> Any:
        """
        Load a batch of segmentation data.
        
        Args:
            input_path (Path): Path to the saved data.
            start (int): Start index of the batch.
            end (int): End index of the batch.
            
        Returns:
            Any: Batch of segmentation data.
        """
        raise NotImplementedError("Subclasses must implement load_segmentation_batch")


class ZarrSegmentationStorage(SegmentationStorage):
    """
    Segmentation storage adapter using Zarr format.
    
    This class handles storing and retrieving segmentation data using Zarr,
    which provides efficient chunked storage for multi-dimensional arrays.
    """
    
    def save_segmentation_data(self, data, metadata: Dict[str, Any], output_path: Path) -> Path:
        """
        Save segmentation data and metadata to Zarr storage.
        
        Args:
            data: Segmentation data as numpy array or xarray DataArray/Dataset.
            metadata (Dict[str, Any]): Metadata about the segmentation.
            output_path (Path): Path to save the data.
            
        Returns:
            Path: Path to the saved Zarr store.
        """
        # Ensure output path has .zarr extension
        zarr_path = output_path.with_suffix('.zarr')
        
        # Convert numpy arrays to xarray if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                # Assuming shape (frames, height, width)
                frames, height, width = data.shape
                data = xr.DataArray(
                    data,
                    dims=["time", "y", "x"],
                    coords={
                        "time": np.arange(frames),
                        "y": np.arange(height),
                        "x": np.arange(width)
                    }
                )
            else:
                raise ValueError(f"Unsupported array shape: {data.shape}")
        
        # If it's a DataArray, convert to Dataset
        if isinstance(data, xr.DataArray):
            data = xr.Dataset({"segmentation": data})
        
        # Add metadata as attributes
        if isinstance(data, xr.Dataset):
            for key, value in metadata.items():
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                # Convert dictionaries with integer keys to string keys for JSON serialization
                elif isinstance(value, dict) and any(isinstance(k, int) for k in value.keys()):
                    value = {str(k): v for k, v in value.items()}
                
                data.attrs[key] = value
        
        # Determine optimal chunking for video data
        if isinstance(data, xr.Dataset) and 'segmentation' in data:
            time_chunks = min(100, data.sizes.get('time', 1))
            y_chunks = min(256, data.sizes.get('y', 1))
            x_chunks = min(256, data.sizes.get('x', 1))
            chunked_data = data.chunk({'time': time_chunks, 'y': y_chunks, 'x': x_chunks})
        else:
            chunked_data = data
            
        # Save to Zarr format with compression
        encoding = {'segmentation': {'compressor': zarr.Blosc(cname='zstd', clevel=3)}}
        chunked_data.to_zarr(zarr_path, mode='w', encoding=encoding)
        
        logger.info(f"Saved segmentation data to {zarr_path}")
        return zarr_path
    
    def load_segmentation_data(self, input_path: Path) -> Tuple[xr.Dataset, Dict[str, Any]]:
        """
        Load segmentation data and metadata from Zarr storage.
        
        Args:
            input_path (Path): Path to the Zarr store.
            
        Returns:
            Tuple[xr.Dataset, Dict[str, Any]]: Tuple of segmentation dataset and metadata.
        """
        # Ensure input path has .zarr extension
        zarr_path = input_path.with_suffix('.zarr')
        
        # Load the dataset
        dataset = xr.open_zarr(zarr_path)
        
        # Extract metadata from attributes
        metadata = dict(dataset.attrs)
        
        return dataset, metadata
    
    def load_segmentation_batch(self, input_path: Path, start: int, end: int) -> xr.Dataset:
        """
        Load a batch of segmentation data from Zarr storage.
        
        Args:
            input_path (Path): Path to the Zarr store.
            start (int): Start index of the batch.
            end (int): End index of the batch.
            
        Returns:
            xr.Dataset: Batch of segmentation data.
        """
        # Ensure input path has .zarr extension
        zarr_path = input_path.with_suffix('.zarr')
        
        # Open the dataset
        dataset = xr.open_zarr(zarr_path)
        
        # Extract the specified time slice
        batch = dataset.isel(time=slice(start, end))
        
        return batch


class ParquetAnalysisStorage:
    """
    Analysis results storage adapter using Parquet format.
    
    This class handles storing and retrieving analysis results using Parquet,
    which provides efficient columnar storage for tabular data.
    """
    
    def save_category_analysis(
        self, 
        counts: Dict[int, int],
        percentages: Dict[int, float],
        output_path: Path,
        frame_idx: Optional[int] = None
    ) -> Path:
        """
        Save category counts and percentages to Parquet storage.
        
        Args:
            counts (Dict[int, int]): Dictionary mapping category IDs to pixel counts.
            percentages (Dict[int, float]): Dictionary mapping category IDs to percentages.
            output_path (Path): Path to save the data.
            frame_idx (Optional[int]): Frame index, for video analysis.
            
        Returns:
            Path: Path to the saved Parquet file.
        """
        # Create a DataFrame with category analysis
        data = []
        for category_id in counts.keys():
            row = {
                'category_id': category_id,
                'pixel_count': counts[category_id],
                'percentage': percentages[category_id]
            }
            if frame_idx is not None:
                row['frame_idx'] = frame_idx
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Ensure output path has .parquet extension
        parquet_path = output_path.with_suffix('.parquet')
        
        # Save to Parquet format
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved category analysis to {parquet_path}")
        return parquet_path
    
    def save_video_analysis(
        self,
        segmentation_data: Union[np.ndarray, xr.DataArray, xr.Dataset],
        metadata: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """
        Save video analysis results to Parquet storage.
        
        Args:
            segmentation_data: Segmentation data.
            metadata (Dict[str, Any]): Metadata about the segmentation.
            output_path (Path): Path to save the data.
            
        Returns:
            Path: Path to the saved Parquet file.
        """
        # Initialize list to hold all frame statistics
        all_stats = []
        
        # Get the appropriate data array
        if isinstance(segmentation_data, xr.Dataset) and 'segmentation' in segmentation_data:
            data_array = segmentation_data.segmentation
        elif isinstance(segmentation_data, xr.DataArray):
            data_array = segmentation_data
        elif isinstance(segmentation_data, np.ndarray):
            # Assuming shape (frames, height, width)
            data_array = segmentation_data
        else:
            raise ValueError("Unsupported data type for segmentation_data")
        
        # Analyze each frame
        if isinstance(data_array, (xr.DataArray, xr.Dataset)):
            frame_count = data_array.sizes.get('time', 1)
            for frame_idx in range(frame_count):
                frame = data_array.isel(time=frame_idx).values if hasattr(data_array, 'isel') else data_array[frame_idx]
                frame_stats = self._analyze_frame(frame, frame_idx)
                all_stats.extend(frame_stats)
        else:
            # Numpy array
            for frame_idx, frame in enumerate(data_array):
                frame_stats = self._analyze_frame(frame, frame_idx)
                all_stats.extend(frame_stats)
        
        # Create DataFrame
        df = pd.DataFrame(all_stats)
        
        # Ensure output path has .parquet extension
        parquet_path = output_path.with_suffix('.parquet')
        
        # Save to Parquet format
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved video analysis to {parquet_path}")
        return parquet_path
    
    def _analyze_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """
        Analyze a single frame and return statistics.
        
        Args:
            frame (np.ndarray): Frame data.
            frame_idx (int): Frame index.
            
        Returns:
            List[Dict[str, Any]]: List of statistics dictionaries.
        """
        # Count pixels per category
        unique_values, counts = np.unique(frame, return_counts=True)
        total_pixels = frame.size
        
        # Create statistics for each category
        frame_stats = []
        for value, count in zip(unique_values, counts):
            percentage = (count / total_pixels) * 100
            frame_stats.append({
                'frame_idx': frame_idx,
                'category_id': int(value),
                'pixel_count': int(count),
                'percentage': float(percentage)
            })
        
        return frame_stats
    
    def load_category_analysis(self, input_path: Path) -> pd.DataFrame:
        """
        Load category analysis from Parquet storage.
        
        Args:
            input_path (Path): Path to the Parquet file.
            
        Returns:
            pd.DataFrame: DataFrame with category analysis.
        """
        # Ensure input path has .parquet extension
        parquet_path = input_path.with_suffix('.parquet')
        
        # Load the DataFrame
        df = pd.read_parquet(parquet_path)
        
        return df


class StorageFactory:
    """
    Factory class for creating storage adapters.
    
    This class provides methods for creating appropriate storage adapters
    based on configuration options.
    """
    
    @staticmethod
    def create_segmentation_storage(config: Config) -> SegmentationStorage:
        """
        Create a segmentation storage adapter based on configuration.
        
        Args:
            config (Config): Configuration object.
            
        Returns:
            SegmentationStorage: Appropriate segmentation storage adapter.
        """
        # For now, we're using Zarr for all segmentation storage
        return ZarrSegmentationStorage()
    
    @staticmethod
    def create_analysis_storage(config: Config) -> ParquetAnalysisStorage:
        """
        Create an analysis storage adapter based on configuration.
        
        Args:
            config (Config): Configuration object.
            
        Returns:
            ParquetAnalysisStorage: Appropriate analysis storage adapter.
        """
        return ParquetAnalysisStorage()
"""
This module provides a core implementation of the CitySeg workflow without Hamilton.

It defines classes for processing images and videos through the segmentation pipeline,
with functionality for extracting frames, applying segmentation, and saving results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
from PIL import Image
from loguru import logger

from .config import Config, InputType, ModelConfig
from .video_processor import VideoProcessor
from .image_processor import ImageProcessor
from .segmentation_processor import SegmentationProcessor
from .dataset_builder import DatasetBuilder
from .visualization_handler import VisualizationHandler


class CitysegCoreWorkflow:
    """
    Core workflow for the CitySeg pipeline without Hamilton dependencies.
    
    This class provides methods for processing images and videos through the
    segmentation pipeline, with automatic handling of different input types.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the workflow with the provided configuration.
        
        Args:
            config: Configuration for the workflow
        """
        self.config = config
    
    def process_video(self) -> Dict[str, Any]:
        """
        Process a video through the segmentation pipeline.
        
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Processing video: {self.config.input}")
            
            # 1. Get video metadata
            video_metadata = VideoProcessor.get_metadata(self.config.input)
            
            # 2. Determine frame indices based on frame step
            frame_indices = VideoProcessor.get_frame_indices(
                video_metadata['frame_count'], 
                self.config.frame_step
            )
            
            # 3. Get video frames
            frames = VideoProcessor.get_frames(self.config.input, frame_indices)
            
            # 4. Create segmentation pipeline
            pipeline = SegmentationProcessor.create_pipeline(self.config.model)
            
            # 5. Process frames through pipeline
            logger.info(f"Processing {len(frames)} frames through segmentation pipeline")
            results = SegmentationProcessor.process_batch(frames, pipeline)
            
            # 6. Extract segmentation maps and metadata
            seg_maps = SegmentationProcessor.extract_segmentation_maps(results)
            seg_metadata = SegmentationProcessor.extract_metadata(results)
            
            # 7. Create dataset
            dataset = DatasetBuilder.create_video_dataset(
                seg_maps,
                video_metadata,
                frame_indices,
                self.config.model.to_dict(),
                seg_metadata
            )
            
            # 8. Save dataset
            output_path = self.config.get_output_path()
            save_path = DatasetBuilder.save_segmentation(dataset, output_path)
            
            # 9. Create analysis if requested
            analysis_path = None
            if self.config.analyze_results:
                analysis_path = DatasetBuilder.save_analysis(dataset, output_path)
            
            # 10. Return results
            result = {
                'segmentation_dataset': dataset,
                'segmentation_path': str(save_path),
            }
            
            if analysis_path:
                result['analysis_path'] = str(analysis_path)
                
            logger.info(f"Processing complete, results saved to: {save_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def process_image(self) -> Dict[str, Any]:
        """
        Process an image through the segmentation pipeline.
        
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Processing image: {self.config.input}")
            
            # 1. Load and preprocess image
            image = ImageProcessor.load_image(self.config.input)
            if self.config.model.max_size:
                image = ImageProcessor.resize_image(image, self.config.model.max_size)
            
            # 2. Create segmentation pipeline
            pipeline = SegmentationProcessor.create_pipeline(self.config.model)
            
            # 3. Process image through pipeline
            logger.info("Processing image through segmentation pipeline")
            result = SegmentationProcessor.process_image(image, pipeline)
            seg_map = result["seg_map"]
            
            # 4. Extract metadata
            seg_metadata = {
                "label2id": result.get("label2id", {}),
                "id2label": result.get("id2label", {}),
                "palette": result.get("palette", None)
            }
            
            # 5. Prepare outputs
            output_path = self.config.get_output_path()
            results = {}
            
            # 6. Create and save visualizations
            visualizer = VisualizationHandler()
            
            if self.config.save_raw_segmentation:
                raw_path = output_path.with_name(f"{output_path.stem}_raw_segmentation.png")
                ImageProcessor.save_image(seg_map.astype(np.uint8), raw_path)
                results["raw_segmentation"] = str(raw_path)
                logger.info(f"Raw segmentation saved to {raw_path}")
            
            if self.config.save_colored_segmentation:
                colored_seg = visualizer.visualize_segmentation(
                    np.array(image), seg_map, seg_metadata['palette'], colored_only=True
                )
                colored_path = output_path.with_name(f"{output_path.stem}_colored_segmentation.png")
                ImageProcessor.save_image(colored_seg, colored_path)
                results["colored_segmentation"] = str(colored_path)
                logger.info(f"Colored segmentation saved to {colored_path}")
            
            if self.config.save_overlay:
                overlay = visualizer.visualize_segmentation(
                    np.array(image), seg_map, seg_metadata['palette'], colored_only=False
                )
                overlay_path = output_path.with_name(f"{output_path.stem}_overlay.png")
                ImageProcessor.save_image(overlay, overlay_path)
                results["overlay"] = str(overlay_path)
                logger.info(f"Overlay saved to {overlay_path}")
            
            # 7. Analyze results if requested
            if self.config.analyze_results:
                from .segmentation_analyzer import SegmentationAnalyzer
                
                # Get the number of categories
                num_categories = len(seg_metadata.get("id2label", {}))
                
                # Analyze the segmentation map
                analysis = SegmentationAnalyzer().analyze_segmentation_map(seg_map, num_categories)
                
                # Extract counts and percentages
                counts = {category_id: count for category_id, (count, _) in analysis.items()}
                percentages = {category_id: percentage for category_id, (_, percentage) in analysis.items()}
                
                # Save analysis
                from .storage_adapter import ParquetAnalysisStorage
                storage = ParquetAnalysisStorage()
                parquet_path = storage.save_category_analysis(
                    counts,
                    percentages,
                    output_path.with_name(f"{output_path.stem}_category_analysis")
                )
                results["analysis"] = str(parquet_path)
                logger.info(f"Category analysis saved to {parquet_path}")
            
            logger.info("Image processing complete")
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def process_directory(self) -> Dict[str, Dict[str, Any]]:
        """
        Process a directory of images or videos.
        
        Returns:
            Dictionary containing results for each processed file
        """
        input_dir = Path(self.config.input)
        
        # Find video files
        video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mov"))
        
        # Find image files
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + \
                     list(input_dir.glob("*.png")) + list(input_dir.glob("*.bmp"))
        
        results = {
            'videos': {},
            'images': {}
        }
        
        # Process videos
        for video_file in video_files:
            logger.info(f"Processing video: {video_file}")
            
            # Create a new config for this video
            video_config = Config(
                input=video_file,
                model=self.config.model,
                output_dir=self.config.output_dir,
                frame_step=self.config.frame_step,
                analyze_results=self.config.analyze_results
            )
            
            # Create a workflow for this video
            workflow = CitysegCoreWorkflow(video_config)
            
            # Process the video
            results['videos'][str(video_file)] = workflow.process_video()
        
        # Process images
        for image_file in image_files:
            logger.info(f"Processing image: {image_file}")
            
            # Create a new config for this image
            image_config = Config(
                input=image_file,
                model=self.config.model,
                output_dir=self.config.output_dir,
                save_raw_segmentation=self.config.save_raw_segmentation,
                save_colored_segmentation=self.config.save_colored_segmentation,
                save_overlay=self.config.save_overlay,
                analyze_results=self.config.analyze_results
            )
            
            # Create a workflow for this image
            workflow = CitysegCoreWorkflow(image_config)
            
            # Process the image
            results['images'][str(image_file)] = workflow.process_image()
        
        return results
    
    def process(self) -> Dict[str, Any]:
        """
        Process the input based on its type.
        
        Returns:
            Dictionary containing processing results
        """
        if self.config.input_type == InputType.SINGLE_VIDEO:
            return self.process_video()
        elif self.config.input_type == InputType.SINGLE_IMAGE:
            return self.process_image()
        elif self.config.input_type == InputType.DIRECTORY:
            return self.process_directory()
        else:
            raise ValueError(f"Unsupported input type: {self.config.input_type}")


def create_core_workflow(config: Config) -> CitysegCoreWorkflow:
    """
    Create a CitysegCoreWorkflow instance.
    
    Args:
        config: Configuration for the workflow
        
    Returns:
        CitysegCoreWorkflow instance
    """
    return CitysegCoreWorkflow(config)
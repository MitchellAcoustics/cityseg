"""
This module provides a Hamilton driver for the CitySeg pipeline.

It creates and configures Hamilton drivers that orchestrate the execution
of the functions defined in hamilton_functions.py.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger
from hamilton import driver

from .config import Config, InputType
from .image_processor import ImageProcessor


def create_video_driver(config: Config, cache_dir: Optional[Path] = None) -> driver.Driver:
    """
    Create a Hamilton driver for video processing.
    
    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results
        
    Returns:
        Configured Hamilton driver for video processing
    """
    # Import hamilton_functions module
    import cityseg.hamilton_functions as hf
    
    # Create initial inputs with all necessary parameters
    initial_inputs = {
        'video_path': str(config.input),
        'frame_step': config.frame_step,
        'output_path': str(config.get_output_path()),
        'model_metadata': config.model.to_dict(),
        'model_name': config.model.name,
        'model_type': config.model.model_type,
        'model_device': config.model.device,
        'model_max_size': config.model.max_size,
        'model_num_workers': config.model.num_workers
    }
    
    # Set up driver builder
    builder = driver.Builder()
    builder = builder.with_modules(hf)
    
    # Add caching if provided
    if cache_dir:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            builder = builder.with_cache(path=str(cache_dir))
            logger.info(f"Hamilton caching enabled at {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to set up caching: {str(e)}")
    
    # Enable parallel execution
    try:
        num_workers = max(1, min(8, config.model.num_workers or 2))
        builder = builder.with_parallel_execution(max_workers=num_workers)
        logger.info(f"Hamilton parallel execution enabled with {num_workers} workers")
    except Exception as e:
        logger.warning(f"Failed to set up parallel execution: {str(e)}")
    
    # Build the driver with config from initial_inputs
    driver_instance = builder.build()
    return driver_instance


def create_image_driver(config: Config, cache_dir: Optional[Path] = None) -> driver.Driver:
    """
    Create a Hamilton driver for image processing.
    
    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results
        
    Returns:
        Configured Hamilton driver for image processing
    """
    # Import hamilton_functions module
    import cityseg.hamilton_functions as hf
    
    # Create initial inputs with all necessary parameters
    initial_inputs = {
        'image_path': str(config.input),
        'output_path': str(config.get_output_path()),
        'model_metadata': config.model.to_dict(),
        'max_size': config.model.max_size,
        'model_name': config.model.name,
        'model_type': config.model.model_type,
        'model_device': config.model.device,
        'model_max_size': config.model.max_size,
        'model_num_workers': config.model.num_workers
    }
    
    # Set up driver builder
    builder = driver.Builder()
    builder = builder.with_modules(hf)
    
    # Add caching if provided
    if cache_dir:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            builder = builder.with_cache(path=str(cache_dir))
            logger.info(f"Hamilton caching enabled at {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to set up caching: {str(e)}")
    
    # Build the driver with config from initial_inputs
    driver_instance = builder.build()
    return driver_instance


def process_video(config: Config, cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Process a video using the Hamilton pipeline.
    
    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results
        
    Returns:
        Dictionary containing processing results
    """
    try:
        logger.info(f"Processing video: {config.input}")
        
        # Create Hamilton driver
        hamilton_driver = create_video_driver(config, cache_dir)
        
        # Define outputs to compute
        outputs = [
            'video_metadata',
            'video_segmentation_dataset',
            'saved_segmentation_path',
        ]
        
        if config.analyze_results:
            outputs.append('saved_analysis_path')
        
        # Create initial inputs
        initial_inputs = {
            'video_path': str(config.input),
            'frame_step': config.frame_step,
            'output_path': str(config.get_output_path()),
            'model_metadata': config.model.to_dict(),
            'model_name': config.model.name,
            'model_type': config.model.model_type,
            'model_device': config.model.device,
            'model_max_size': config.model.max_size,
            'model_num_workers': config.model.num_workers
        }
        
        # Execute the driver to compute outputs
        results = hamilton_driver.execute(outputs, inputs=initial_inputs)
        
        # Extract and return results
        return {
            'metadata': results['video_metadata'],
            'dataset': results['video_segmentation_dataset'],
            'segmentation_path': results['saved_segmentation_path'],
            'analysis_path': results.get('saved_analysis_path')
        }
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def process_image(config: Config, cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Process an image using the Hamilton pipeline.
    
    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results
        
    Returns:
        Dictionary containing processing results
    """
    try:
        logger.info(f"Processing image: {config.input}")
        
        # Create Hamilton driver
        hamilton_driver = create_image_driver(config, cache_dir)
        
        # Define outputs to compute
        outputs = [
            'image_data',
            'image_segmentation_result',
            'segmentation_map',
            'single_segmentation_metadata',
        ]
        
        # Add visualization outputs based on configuration
        visualization_outputs = {}
        
        if config.save_raw_segmentation:
            outputs.append('segmentation_map')  # Already included above
            visualization_outputs['raw_segmentation'] = ('segmentation_map', 'raw_segmentation')
        
        if config.save_colored_segmentation:
            outputs.append('colored_segmentation')
            visualization_outputs['colored_segmentation'] = ('colored_segmentation', 'colored_segmentation')
        
        if config.save_overlay:
            outputs.append('segmentation_overlay')
            visualization_outputs['overlay'] = ('segmentation_overlay', 'overlay')
        
        # Add analysis outputs if requested
        if config.analyze_results:
            outputs.append('category_analysis')
            outputs.append('saved_category_analysis_path')
        
        # Create initial inputs
        initial_inputs = {
            'video_path': str(config.input),
            'frame_step': config.frame_step,
            'output_path': str(config.get_output_path()),
            'model_metadata': config.model.to_dict(),
            'model_name': config.model.name,
            'model_type': config.model.model_type,
            'model_device': config.model.device,
            'model_max_size': config.model.max_size,
            'model_num_workers': config.model.num_workers
        }
        
        # Execute the driver to compute outputs
        results = hamilton_driver.execute(outputs, inputs=initial_inputs)
        
        # Save visualizations
        visualization_paths = {}
        for key, (output_name, suffix) in visualization_outputs.items():
            if output_name in results:
                # Create a visualization path using saved_visualization_path function
                vis_path = str(ImageProcessor.save_image(
                    results[output_name],
                    Path(config.get_output_path()).with_name(f"{Path(config.get_output_path()).stem}_{suffix}.png")
                ))
                visualization_paths[key] = vis_path
        
        # Extract and return results
        return {
            'image': results['image_data'],
            'segmentation_result': results['image_segmentation_result'],
            'segmentation_map': results['segmentation_map'],
            'metadata': results['single_segmentation_metadata'],
            'visualization_paths': visualization_paths,
            'analysis_path': results.get('saved_category_analysis_path')
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def process_directory(config: Config, cache_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Process a directory of images or videos.
    
    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results
        
    Returns:
        Dictionary containing processing results for each file
    """
    from pathlib import Path
    
    input_dir = Path(config.input)
    
    # Find video files
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mov"))
    
    # Find image files
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + \
                 list(input_dir.glob("*.png")) + list(input_dir.glob("*.bmp"))
    
    results = {
        'videos': {},
        'images': {}
    }
    
    # Process video files
    for video_file in video_files:
        logger.info(f"Processing video: {video_file}")
        
        # Create a new config for this video
        video_config = Config(
            input=video_file,
            model=config.model,
            output_dir=config.output_dir,
            frame_step=config.frame_step,
            analyze_results=config.analyze_results
        )
        
        # Process the video
        results['videos'][str(video_file)] = process_video(video_config, cache_dir)
    
    # Process image files
    for image_file in image_files:
        logger.info(f"Processing image: {image_file}")
        
        # Create a new config for this image
        image_config = Config(
            input=image_file,
            model=config.model,
            output_dir=config.output_dir,
            save_raw_segmentation=config.save_raw_segmentation,
            save_colored_segmentation=config.save_colored_segmentation,
            save_overlay=config.save_overlay,
            analyze_results=config.analyze_results
        )
        
        # Process the image
        results['images'][str(image_file)] = process_image(image_config, cache_dir)
    
    return results


def process(config: Config, cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Process input based on its type using the Hamilton pipeline.
    
    Args:
        config: Configuration for the processing pipeline
        cache_dir: Optional directory for caching results
        
    Returns:
        Dictionary containing processing results
    """
    if config.input_type == InputType.SINGLE_VIDEO:
        return process_video(config, cache_dir)
    elif config.input_type == InputType.SINGLE_IMAGE:
        return process_image(config, cache_dir)
    elif config.input_type == InputType.DIRECTORY:
        return process_directory(config, cache_dir)
    else:
        raise ValueError(f"Unsupported input type: {config.input_type}")
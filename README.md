# Semantic Segmentation Pipeline

This project implements a flexible and efficient semantic segmentation pipeline for processing images and videos. It supports multiple segmentation models and datasets, with capabilities for tiling large images, mixed-precision processing, and comprehensive result analysis.

## Features

- Support for multiple segmentation models (BEIT, OneFormer)
- Compatible with various datasets (Cityscapes, ADE20k)
- Flexible image resizing and tiling for processing high-resolution inputs
- Mixed-precision processing for improved performance
- Comprehensive analysis of segmentation results, including category-wise statistics
- Support for both image and video inputs
- Multi-video processing capability for entire directories
- Parallel processing of videos for improved efficiency
- Caching of processed segmentation maps in HDF5 format for quick re-analysis
- Output includes segmentation maps, overlay visualizations, and detailed CSV reports

## Project Structure

The project is organized into several Python modules:

- `main.py`: Entry point of the application, handles argument parsing and high-level flow
- `config.py`: Defines configuration classes for the pipeline
- `models.py`: Implements the segmentation model classes
- `processors.py`: Contains classes for processing images, videos, and directories
- `utils.py`: Provides utility functions for analysis and file operations
- `palettes.py`: Defines color palettes for different datasets

## Dependencies

Ensure you have the following packages installed:

- PyTorch
- torchvision
- transformers
- opencv-python (cv2)
- numpy
- Pillow (PIL)
- h5py
- pandas
- tqdm
- pyyaml

You can install these dependencies using pip:

```
pip install torch torchvision transformers opencv-python numpy Pillow h5py pandas tqdm pyyaml
```

## Usage

1. Prepare a configuration YAML file (e.g., `config.yaml`) with your desired settings.

2. Run the pipeline using the following command:

   ```
   python main.py --config path/to/your/config.yaml
   ```

   Optional arguments:
   - `--input`: Path to input image, video, or directory (overrides config file)
   - `--output`: Path to output prefix (overrides config file)
   - `--frame_step`: Process every nth frame for videos (overrides config file)
   - `--num_workers`: Number of parallel workers for directory processing (overrides config file)

3. The pipeline will process the input and generate the following outputs:
   - Segmentation maps (as HDF5 file)
   - Colored segmentation visualizations
   - Overlay of segmentation on original input
   - CSV files with frame-by-frame category counts and percentages
   - CSV files with overall category statistics

## Configuration

The `config.yaml` file should include the following sections:

```yaml
input: path/to/your/input/file_or_directory
output_prefix: path/to/your/output/directory/output
model:
  type: oneformer  # or beit
  name: shi-labs/oneformer_ade20k_swin_large
  dataset: ade20k  # or cityscapes
  max_size: 1920  # Set to null to maintain original resolution
  tile_size: 960  # Set to null to disable tiling
  mixed_precision: true
frame_step: 5  # For video processing, process every 5th frame
num_workers: 4  # Number of parallel workers for directory processing
```

## Multi-Video Processing

To process multiple videos:

1. Set the `input` in your configuration file to a directory containing video files.
2. Specify the number of parallel workers using the `num_workers` parameter.
3. Run the pipeline as usual.

The pipeline will process all supported video files in the directory, saving results in separate subdirectories for each video.

## Caching and Re-analysis

The pipeline saves processed segmentation maps in HDF5 format. If you run the pipeline on a previously processed video, it will detect the existing HDF5 file and perform only the analysis step, saving time on re-processing.

## Extending the Pipeline

To add support for new segmentation models:

1. Implement a new class in `models.py` that inherits from `SegmentationModelBase`
2. Add the new model type to the `create_segmentation_model` function in `models.py`
3. Update the configuration handling in `config.py` if necessary

## License

[Specify your license here]

## Contributing

[Provide guidelines for contributing to the project]

## Contact

[Your contact information or how to reach out for support]
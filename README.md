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

**A note on the `natten` package**: 

To use the `dinat` backbone for OneDormer models, you need to have `natten` installed. Unfortunately, the current version has some installation issues, at least on MacOS. `pip` fails to recognise the installed `torch` package, which is required for `natten`. If you encounter problems, you can try the following steps, based on [this issue response](https://github.com/SHI-Labs/NATTEN/issues/90#issuecomment-2027235265):

1. In your chosen interpreter, find the location of the `torch` package by running `import torch; print(torch.__file__)`. For example, it might be in a path like `/Users/username/Documents/GitHub/cityscape-seg/.venv/lib/python3.12/site-packages/torch/__init__.py`.

2. Set the environment variable `PYTHONPATH` to that path but excluding the `/torch/__init__.py`:
3. Install `natten` using `pip` (or `rye add` if you are using `rye`):
4. Unset the `PYTHONPATH` variable:

```bash
export PYTHONPATH=/Users/username/Documents/GitHub/cityscape-seg/.venv/lib/python3.12/site-packages

pip install natten

unset PYTHONPATH
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

## Models

In theory, the pipeline supports two segmentation models: BEIT and OneFormer (in practice only OneFormer works properly at the moment). You can specify the model type, name, dataset, and other parameters in the configuration file.

The verified models are:

- OneFormer: `shi-labs/oneformer_ade20k_swin_large`
- OneFormer: `shi-labs/oneformer_cityscapes_swin_large`
- OneFormer: `shi-labs/oneformer_ade20k_dinat_large`
- OneFormer: `shi-labs/oneformer_cityscapes_dinat_large`

**However, please note that the `dinat` models have some quirks.** First, `natten`, which is required for the `dinat` backbone does not work on MPS, so we force it to use the CPU. Second, they are _much_ slower than the `swin` backbone models, like 100x slower. Partly this is due to running on the cpu, but I also suspect they are just slower. However, they seem to have better outputs. I will be exploring this further.

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
  device: mps # or cpu or cuda
frame_step: 5  # For video processing, process every 5th frame
save_raw_segmentation: true # Save raw segmentation maps in HDF5 format
save_colored_segmentation: false # Save colored segmentation maps as mp4
save_overlay: true # Save overlay visualizations as mp4
visualization:
  alpha: 0.5
  colormap: default
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
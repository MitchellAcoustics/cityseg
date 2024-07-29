# CitySeg: Urban Semantic Segmentation Pipeline

CitySeg is a flexible and efficient semantic segmentation pipeline for processing images and videos of urban environments. It supports multiple segmentation models and datasets, with capabilities for processing high-resolution inputs and comprehensive result analysis.

## Features

- Support for multiple segmentation models (OneFormer)
- Compatible with various datasets (Cityscapes, ADE20k, Mapillary Vistas)
- Flexible image resizing for processing high-resolution inputs
- Comprehensive analysis of segmentation results, including category-wise statistics
- Support for both image and video inputs
- Multi-video processing capability for entire directories
- Caching of processed segmentation maps in HDF5 format for quick re-analysis
- Output includes segmentation maps, colored segmentations, overlay visualizations, and detailed CSV reports
- Configurable logging with console and file outputs

## Installation

You can install CitySeg using pip:

```
pip install cityseg
```

Alternatively, clone the repository and install the dependencies:

```
git clone https://github.com/your-username/cityseg.git
cd cityseg
pip install -e .
```

## Dependencies

CitySeg requires the following main packages:

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
- loguru

For a complete list of dependencies, please refer to the `pyproject.toml` file.

## Usage

1. Prepare a configuration YAML file (e.g., `config.yaml`) with your desired settings.

2. Run the pipeline using the following command:

   ```
   python -m cityseg.main --config path/to/your/config.yaml
   ```

   Optional arguments:
   - `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - `--verbose`: Enable verbose logging

3. The pipeline will process the input and generate various outputs including segmentation maps, visualizations, and analysis reports.

## Configuration

Here's an example of a basic configuration file:

```yaml
input: path/to/your/input/file_or_directory
output_prefix: path/to/your/output/directory/output
model:
  name: shi-labs/oneformer_cityscapes_swin_large
  max_size: 1920  # Set to null to maintain original resolution
  device: cuda  # or cpu or mps
frame_step: 5  # For video processing, process every 5th frame
save_raw_segmentation: true
save_colored_segmentation: true
save_overlay: true
visualization:
  alpha: 0.5
  colormap: default
```

## Models

CitySeg currently supports Mask2Former and BEIT models. The verified models include:

- "facebook/mask2former-swin-large-cityscapes-semantic"
- "facebook/mask2former-swin-large-mapillary-vistas-semantic"
- "facebook/maskformer-swin-small-ade" (sort of, this often leads to segfaults. Recommend using `disable_tqdm` in the config.)
- "microsoft/beit-large-finetuned-ade-640-640"
- "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
- "zoheb/mit-b5-finetuned-sidewalk-semantic"
- ""nickmuchi/segformer-b4-finetuned-segments-sidewalk""

Mask2Former are by far the most stable.

Some models which seem to load correctly but continually produce segfault errors on my machine are:

- "facebook/maskformer-swin-large-ade"
- "nvidia/segformer-b5-finetuned-ade-640-640"
- "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
- "zoheb/mit-b5-finetuned-sidewalk-semantic" (use `model_type: segformer` in the config)

Confirmed not to work due to issues with the Hugging Face pipeline:

- "shi-labs/oneformer_ade20k_dinat_large"

**Note on `dinat` models:** The `dinat` backbone models require the `natten` package, which may have installation issues on some systems. These models are also significantly slower than the `swin` backbone models, especially when forced to run on CPU. However, they may produce better quality outputs in some cases.

## Project Structure

The project is organized into several Python modules, each serving a specific purpose within the CitySeg pipeline:

- `main.py`: Entry point of the application, responsible for initializing and running the segmentation pipeline.
- `config.py`: Defines configuration classes and handles loading and validating configuration settings.
- `pipeline.py`: Implements the core segmentation pipeline, including model loading and inference.
- `processors.py`: Contains classes for processing images, videos, and directories, managing the segmentation workflow.
- `segmentation_analyzer.py`: Provides functionality for analyzing segmentation results, including computing statistics and generating reports.
- `video_file_iterator.py`: Implements an iterator for efficiently processing multiple video files in a directory.
- `visualization_handler.py`: Handles the visualization of segmentation results using color palettes.
- `file_handler.py`: Manages file operations related to saving and loading segmentation data and metadata.
- `utils.py`: Provides utility functions for various tasks, including data handling and logging.
- `palettes.py`: Defines color palettes for different datasets used in segmentation.
- `exceptions.py`: Custom exception classes for error handling throughout the pipeline.

This modular structure allows for easy maintenance and extension of the CitySeg pipeline, facilitating the addition of new features and models.

## Logging

The pipeline uses the `loguru` library for flexible and configurable logging. You can set the log level and enable verbose output using command-line arguments:

```
python main.py --config path/to/your/config.yaml --log-level INFO # or DEBUG, WARNING, ERROR, CRITICAL or --verbose
```

Logs are output to both the console and a file (`segmentation.log`). The file log is in JSON format for easy parsing and analysis.


## Extending CitySeg

To add support for new segmentation models:

1. Implement a new model class in `pipeline.py`
2. Update the `create_segmentation_pipeline` function in `pipeline.py`
3. Update the configuration handling in `config.py` if necessary

## Contributing

Contributions to CitySeg are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

## License

CitySeg is released under the BSD 3-Clause License. See the `LICENSE` file for details.

## Contact

For support or inquiries, please open an issue on the GitHub repository or contact Andrew Mitchell.
import argparse
import logging
import sys
from pathlib import Path

from .config import Config
from .models import create_segmentation_model
from .processors import create_processor


def setup_logging():
    """
    Set up logging configuration for the application.

    This function configures the logging system to output log messages
    to both the console and a file named 'segmentation.log'. It sets
    the logging level to INFO and includes timestamp, log level,
    logger name, file name, line number, and the log message in the output.

    Returns:
        logging.Logger: Configured logger object.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("segmentation.log"),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def main():
    """
    Main entry point for the semantic segmentation application.

    This function parses command-line arguments, loads the configuration,
    initializes the segmentation model and processor, and executes the
    segmentation process. It handles both image and video inputs,
    saving the results and analysis data to specified output locations.

    The function uses a try-except block to catch and log any errors
    that occur during the process.

    Command-line arguments:
        --config: Path to the configuration YAML file (required)
        --input: Path to input image or video (overrides config file)
        --output: Path to output prefix (overrides config file)
        --frame_step: Process every nth frame (overrides config file)

    Returns:
        None
    """

    parser = argparse.ArgumentParser(
        description="Semantic Segmentation for High-Resolution Images and Videos"
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input image, video, or directory (overrides config file)",
    )
    parser.add_argument(
        "--output", type=str, help="Path to output prefix (overrides config file)"
    )
    parser.add_argument(
        "--frame_step", type=int, help="Process every nth frame (overrides config file)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of parallel workers for directory processing (overrides config file)",
    )
    args = parser.parse_args()

    try:
        config = Config.from_yaml(Path(args.config))

        # Override config with command-line arguments
        if args.input:
            config.input = Path(args.input)
        if args.frame_step:
            config.frame_step = args.frame_step
        if args.output:
            config.output_prefix = Path(args.output)
        if args.num_workers:
            config.num_workers = args.num_workers

        # Input validation
        if not Path(config.input).exists():
            raise FileNotFoundError(f"Input path not found: {config.input}")

        logger.info(f"Configuration loaded: {config.to_dict()}")

        processor = create_processor(config.to_dict())
        processor.process()

    except Exception as e:
        logger.exception(f"An error occurred during processing: {str(e)}")
    finally:
        logger.info("Processing complete.")


if __name__ == "__main__":
    main()

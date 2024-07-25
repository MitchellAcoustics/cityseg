import argparse
import logging
import sys
from pathlib import Path

from cityscape_seg.config import Config
from cityscape_seg.processors import create_processor


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
        "--output_dir",
        type=str,
        help="Path to output directory (overrides config file)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        help="Prefix for output files (overrides config file)",
    )
    parser.add_argument(
        "--frame_step", type=int, help="Process every nth frame (overrides config file)"
    )

    args = parser.parse_args()

    try:
        config = Config.from_yaml(Path(args.config))

        # Override config with command-line arguments
        if args.input:
            config.input = Path(args.input)
        if args.frame_step:
            config.frame_step = args.frame_step
        if args.output_dir:
            config.output_dir = Path(args.output_dir)
        if args.output_prefix:
            config.output_prefix = args.output_prefix

        # Input validation
        if not config.input.exists():
            raise FileNotFoundError(f"Input path not found: {config.input}")

        logger.info(f"Configuration loaded: {config.to_dict()}")

        processor = create_processor(config)
        processor.process()

    except Exception as e:
        logger.exception(f"An error occurred during processing: {str(e)}")
    finally:
        logger.info("Processing complete.")


if __name__ == "__main__":
    main()

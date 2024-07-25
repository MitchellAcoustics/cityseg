import argparse
import warnings
from pathlib import Path

from cityscape_seg.config import Config
from cityscape_seg.exceptions import (
    ConfigurationError,
    InputError,
    ModelError,
    ProcessingError,
)
from cityscape_seg.processors import create_processor
from loguru import logger
from tqdm import tqdm


class TqdmCompatibleSink:
    def __init__(self, compact=True):
        self.compact = compact

    def write(self, message):
        tqdm.write(message, end="")


def setup_logging(log_level, verbose=False):
    logger.remove()  # Remove default handler

    # Console logging
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.add(
        TqdmCompatibleSink(compact=not verbose),
        format=console_format,
        level=log_level,
        colorize=True,
    )

    # File logging (always verbose, JSON format)
    logger.add("segmentation.log", format="{message}", level="DEBUG", serialize=True)


def parse_arguments():
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
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


@logger.catch
def main():
    args = parse_arguments()
    setup_logging(args.log_level, args.verbose)
    logger.info(f"Starting segmentation process with input: {args.input}")

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
            raise InputError(f"Input path not found: {config.input}")

        logger.info("Configuration loaded", config=config.to_dict())

        processor = create_processor(config)
        processor.process()

    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
    except ProcessingError as e:
        logger.error(f"Processing error: {str(e)}")
    except ModelError as e:
        logger.error(f"Model error: {str(e)}")
    except InputError as e:
        logger.error(f"Input error: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
    finally:
        logger.info("Processing complete.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

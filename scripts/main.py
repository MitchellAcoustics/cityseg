import argparse
import logging
from tqdm import tqdm
from pathlib import Path

from cityscape_seg.config import Config
from cityscape_seg.processors import create_processor


def setup_logging(log_level):
    """
    Set up logging configuration for the application.

    Args:
        log_level (str): Desired logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logging.Logger: Configured logger object.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            TqdmLoggingHandler(),
            logging.FileHandler("segmentation.log"),
        ],
    )
    return logging.getLogger(__name__)

def main():
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
        help="Set the logging level"
    )

    args = parser.parse_args()

    logger = setup_logging(args.log_level)

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
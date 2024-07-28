"""
This module serves as the entry point for the Semantic Segmentation Pipeline.

It handles command-line argument parsing, configuration loading, and
orchestrates the overall execution of the segmentation process.
"""

import argparse
from pathlib import Path

from loguru import logger

from cityseg.config import Config
from cityseg.exceptions import (
    ConfigurationError,
    InputError,
    ModelError,
    ProcessingError,
)
from cityseg.processors import create_processor
from cityseg.utils import setup_logging


@logger.catch
def main() -> None:
    """
    Main function to run the Semantic Segmentation Pipeline.

    This function parses command-line arguments, sets up logging,
    loads the configuration, creates the appropriate processor,
    and executes the segmentation process. It also handles and
    logs any exceptions that occur during execution.
    """
    parser = argparse.ArgumentParser(description="Semantic Segmentation Pipeline")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    setup_logging(args.log_level, args.verbose)

    try:
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(Path(args.config))
        logger.info(f"Loaded configuration: {config}")

        logger.info(f"Creating processor for input type: {config.input_type}")
        processor = create_processor(config)
        logger.debug(f"Processor created: {type(processor).__name__}")

        processor.process()
        logger.info("Processing completed successfully")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
    except InputError as e:
        logger.error(f"Input error: {str(e)}")
    except ModelError as e:
        logger.error(f"Model error: {str(e)}")
    except ProcessingError as e:
        logger.error(f"Processing error: {str(e)}")
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    main()

"""
This module serves as the entry point for the Semantic Segmentation Pipeline.

It handles command-line argument parsing, configuration loading, and
orchestrates the overall execution of the segmentation process.
"""

import argparse
from pathlib import Path

from loguru import logger

from cityseg.core import (
    Config,
    ConfigurationError,
    InputError,
    ModelError,
    ProcessingError,
)
from cityseg.utils.common import setup_logging
from cityseg.workflow import process


@logger.catch
def main() -> None:
    """
    Main function to run the Semantic Segmentation Pipeline.

    This function parses command-line arguments, sets up logging,
    loads the configuration, and executes the segmentation process.
    It also handles and logs any exceptions that occur during execution.
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
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to use for caching intermediate results",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable caching of intermediate results",
    )
    args = parser.parse_args()

    setup_logging(args.log_level, args.verbose)

    try:
        logger.info(f"Loading configuration from {args.config}")
        config = Config.from_yaml(Path(args.config))
        logger.info(f"Loaded configuration: {config}")

        # Determine cache directory
        cache_dir = None
        if not args.disable_cache:
            if args.cache_dir:
                cache_dir = Path(args.cache_dir)
            else:
                # Use a default cache directory next to the output
                cache_dir = config.get_output_path().parent / ".cache"
                cache_dir.mkdir(exist_ok=True)

        # Process using Hamilton
        logger.info(f"Processing input type: {config.input_type} with Hamilton")
        result = process(config, cache_dir)
        logger.info(f"Processing completed successfully: {result}")

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

    # import sys
    # sys.argv = ["", "config.yaml", "--verbose"]
    main()

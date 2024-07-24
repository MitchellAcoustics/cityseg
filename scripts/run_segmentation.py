#!/usr/bin/env python3
"""
This script runs the semantic segmentation pipeline on images or videos.

Usage:
    python run_segmentation.py --config path/to/config.yaml
"""

import argparse
import logging
import warnings
from pathlib import Path

from cityscape_seg.config import Config
from cityscape_seg.processors import create_processor

# warnings.filterwarnings("ignore")


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def main():
    """Main entry point for the semantic segmentation script."""
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="Run semantic segmentation on images or videos."
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        processor = create_processor(config.to_dict())
        logger.info(f"Created processor for input: {config.input}")

        processor.process()
        logger.info("Processing completed successfully")

    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

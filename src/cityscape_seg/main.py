# %%
import argparse
from pathlib import Path

from loguru import logger

from cityscape_seg.config import Config
from cityscape_seg.processors import create_processor
from cityscape_seg.exceptions import (
    ConfigurationError,
    InputError,
    ModelError,
    ProcessingError,
)
from cityscape_seg.utils import setup_logging


@logger.catch
def main():
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
        logger.debug(f"Loaded configuration: {config}")

        logger.info(f"Creating processor for input type: {config.input_type}")
        processor = create_processor(config)
        logger.debug(f"Processor created: {type(processor).__name__}")

        logger.info("Starting processing")
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
    # %%
    # from cityscape_seg.config import ModelConfig, Config, VisualizationConfig
    #
    # model_config = ModelConfig(
    #     name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
    #     device="mps",
    # )
    # config = Config(
    #     input=Path(
    #         "/Users/mitch/Documents/GitHub/cityscape-seg/example_inputs/Carlov2_15s_3840x2160.mov"
    #     ),
    #     output_dir=None,
    #     output_prefix=None,
    #     model=model_config,
    #     frame_step=10,
    #     batch_size=45,
    #     save_raw_segmentation=True,
    #     save_colored_segmentation=True,
    #     save_overlay=True,
    #     visualization=VisualizationConfig(alpha=0.5, colormap="default"),
    # )
    # print(config)
    # logger.info(f"Configuration loaded from {config}")
    #
    # processor = create_processor(config)
    # logger.info(f"Created processor for input type: {config.input_type}")
    #
    # processor.process()
    # logger.info("Processing completed successfully")
    import warnings

    warnings.filterwarnings("ignore")

    main()

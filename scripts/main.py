# %%
import argparse
from pathlib import Path

from loguru import logger

from cityscape_seg.config import Config
from cityscape_seg.processors import create_processor
from cityscape_seg.exceptions import ConfigurationError, InputError, ModelError, ProcessingError
from tqdm.auto import tqdm

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

@logger.catch
def main():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Pipeline")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    try:
        config = Config.from_yaml(Path(args.config))
        logger.info(f"Configuration loaded from {args.config}")

        processor = create_processor(config)
        logger.info(f"Created processor for input type: {config.input_type}")

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
    from cityscape_seg.config import ModelConfig, Config, VisualizationConfig
    import warnings

    model_config = ModelConfig(
        name="facebook/mask2former-swin-large-mapillary-vistas-semantic",
            device='mps',
        )
    config = Config(
        input=Path("/Users/mitch/Documents/GitHub/cityscape-seg/example_inputs/Carlov2_15s_3840x2160.mov"),
        output_dir=None,
        output_prefix=None,
        model=model_config,
        frame_step=10,
        batch_size=45,
        save_raw_segmentation=False,
        save_colored_segmentation=True,
        save_overlay=True,
        visualization=VisualizationConfig(alpha=0.5, colormap="default"),
        )
    print(config)
    logger.info(f"Configuration loaded from {config}")

    processor = create_processor(config)
    logger.info(f"Created processor for input type: {config.input_type}")

    processor.process()
    logger.info("Processing completed successfully")

    # warnings.filterwarnings("ignore")
    # main()
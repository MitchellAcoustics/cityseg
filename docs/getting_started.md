# Getting Started

This guide will help you set up and run your first semantic segmentation task using our pipeline.

## Installation

Install the package using pip:

```bash
pip install git+https://github.com/yourusername/cityscape-seg.git
```

## Basic Usage

Here's a simple example to get you started:

```python
from semantic_segmentation import Config, create_segmentation_model, create_processor

# Load configuration
config = Config.from_yaml("path/to/your/config.yaml")

# Create model and processor
model = create_segmentation_model(config.model.to_dict())
processor = create_processor(config.to_dict())

# Process input
processor.process()
```

For more detailed usage examples, please refer to the [Examples](examples.md) section.
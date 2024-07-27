from pathlib import Path

import pytest
import yaml
from cityseg.config import Config, InputType, ModelConfig, VisualizationConfig


@pytest.fixture
def temp_file(tmp_path):
    def _temp_file(filename):
        file_path = tmp_path / filename
        file_path.touch()
        return file_path

    return _temp_file


@pytest.fixture
def temp_directory(tmp_path):
    dir_path = tmp_path / "test_directory"
    dir_path.mkdir()
    return dir_path


@pytest.fixture
def model_config():
    return ModelConfig(
        name="test_model", model_type="oneformer", max_size=1024, device="cuda"
    )


@pytest.fixture
def visualization_config():
    return VisualizationConfig(alpha=0.7, colormap="viridis")


class TestConfig:
    def test_config_determines_input_type_correctly(
        self, temp_file, temp_directory, model_config, tmp_path
    ):
        image_config = Config(
            input=temp_file("test.jpg"),
            output_dir=tmp_path / "output",
            output_prefix="test_prefix",
            model=model_config,
        )
        video_config = Config(
            input=temp_file("test.mp4"),
            output_dir=tmp_path / "output",
            output_prefix="test_prefix",
            model=model_config,
        )
        dir_config = Config(
            input=temp_directory,
            output_dir=tmp_path / "output",
            output_prefix="test_prefix",
            model=model_config,
        )

        assert image_config.input_type == InputType.SINGLE_IMAGE
        assert video_config.input_type == InputType.SINGLE_VIDEO
        assert dir_config.input_type == InputType.DIRECTORY

    def test_config_generates_correct_output_paths(
        self, temp_file, tmp_path, model_config
    ):
        input_path = temp_file("test_video.mp4")
        output_dir = tmp_path / "output"
        config = Config(
            input=input_path,
            output_dir=output_dir,
            output_prefix="test_prefix",
            model=model_config,
        )

        assert config.get_output_path().parent == output_dir
        assert config.get_output_path().name.startswith("test_prefix")
        assert config.get_output_path().suffix == ".mp4"

    def test_config_yaml_roundtrip(self, tmp_path, model_config, visualization_config):
        # Create a dummy video file
        input_file = tmp_path / "test.mp4"
        input_file.touch()

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        original_config = Config(
            input=input_file,
            output_dir=output_dir,
            output_prefix="test_prefix",
            model=model_config,
            visualization=visualization_config,
        )

        # Save to YAML
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(original_config.to_dict(), f)

        # Load from YAML
        loaded_config = Config.from_yaml(yaml_path)

        # Compare
        assert str(loaded_config.input) == str(original_config.input)
        assert str(loaded_config.output_dir) == str(original_config.output_dir)
        assert loaded_config.output_prefix == original_config.output_prefix
        assert loaded_config.model.name == original_config.model.name
        assert loaded_config.visualization.alpha == original_config.visualization.alpha

    @pytest.mark.parametrize("invalid_input", ["nonexistent.txt", "invalid_dir/"])
    def test_config_raises_error_for_invalid_input(
        self, invalid_input, model_config, tmp_path
    ):
        with pytest.raises(ValueError):
            Config(
                input=Path(invalid_input),
                output_dir=tmp_path / "output",
                output_prefix="test_prefix",
                model=model_config,
            )

    # You might keep one simple test for ModelConfig and VisualizationConfig
    def test_model_and_visualization_config_basic(self):
        model_config = ModelConfig(name="test")
        vis_config = VisualizationConfig(alpha=0.5)
        assert model_config.name == "test"
        assert vis_config.alpha == 0.5


class TestDetermineInputType:
    @pytest.fixture
    def model_config(self):
        return ModelConfig(name="test_model")

    @pytest.fixture
    def base_config(self, tmp_path, model_config):
        return lambda input_path: Config(
            input=input_path,
            output_dir=tmp_path / "output",
            output_prefix="test_prefix",
            model=model_config,
        )

    def test_directory_input(self, tmp_path, base_config):
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()
        config = base_config(dir_path)
        assert config.input_type == InputType.DIRECTORY

    @pytest.mark.parametrize("video_ext", [".mp4", ".avi", ".mov"])
    def test_video_input(self, tmp_path, video_ext, base_config):
        video_path = tmp_path / f"test_video{video_ext}"
        video_path.touch()
        config = base_config(video_path)
        assert config.input_type == InputType.SINGLE_VIDEO

    @pytest.mark.parametrize(
        "image_ext", [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    )
    def test_image_input(self, tmp_path, image_ext, base_config):
        image_path = tmp_path / f"test_image{image_ext}"
        image_path.touch()
        config = base_config(image_path)
        assert config.input_type == InputType.SINGLE_IMAGE

    def test_unsupported_input(self, tmp_path, base_config):
        unsupported_path = tmp_path / "unsupported.txt"
        unsupported_path.touch()
        with pytest.raises(
            ValueError, match=f"Unsupported input type: {unsupported_path}"
        ):
            base_config(unsupported_path)

    def test_nonexistent_input(self, tmp_path, base_config):
        nonexistent_path = tmp_path / "nonexistent.jpg"
        with pytest.raises(ValueError):
            base_config(nonexistent_path)

    def test_case_insensitivity(self, tmp_path, base_config):
        upper_case_path = tmp_path / "TEST_IMAGE.JPG"
        upper_case_path.touch()
        config = base_config(upper_case_path)
        assert config.input_type == InputType.SINGLE_IMAGE


class TestGenerateOutputPrefix:
    @pytest.fixture
    def base_config(self, tmp_path):
        def _config(input_path, model_name, frame_step):
            model_config = ModelConfig(name=model_name)
            return Config(
                input=input_path,
                output_dir=tmp_path / "output",
                output_prefix=None,
                model=model_config,
                frame_step=frame_step,
            )

        return _config

    @pytest.mark.parametrize(
        "input_name, model_name, frame_step, expected",
        [
            ("input_dir", "model", 1, "input_dir_model_step1"),
            ("input_dir", "model", 5, "input_dir_model_step5"),
            ("input_dir", "org/model", 1, "input_dir_model_step1"),
            ("input_dir", "org/model", 5, "input_dir_model_step5"),
        ],
    )
    def test_directory_input(
        self, tmp_path, base_config, input_name, model_name, frame_step, expected
    ):
        input_dir = tmp_path / input_name
        input_dir.mkdir()
        config = base_config(input_dir, model_name, frame_step)
        assert config.generate_output_prefix() == expected

    @pytest.mark.parametrize(
        "input_name, model_name, frame_step, expected",
        [
            ("video_input.mp4", "model", 1, "video_model_step1"),
            ("video_input_long_name.mp4", "model", 5, "video_model_step5"),
            ("image_input.jpg", "org/model", 1, "image_model_step1"),
            ("image_input_long_name.png", "org/model", 5, "image_model_step5"),
        ],
    )
    def test_file_input(
        self, tmp_path, base_config, input_name, model_name, frame_step, expected
    ):
        input_file = tmp_path / input_name
        input_file.touch()
        config = base_config(input_file, model_name, frame_step)
        assert config.generate_output_prefix() == expected

    def test_complex_input_name(self, tmp_path, base_config):
        input_file = tmp_path / "complex_name_with_underscores.mp4"
        input_file.touch()
        config = base_config(input_file, "model", 1)
        assert config.generate_output_prefix() == "complex_model_step1"

    def test_model_name_with_multiple_slashes(self, tmp_path, base_config):
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        config = base_config(input_file, "org/suborg/model", 1)
        assert config.generate_output_prefix() == "input_model_step1"

    def test_zero_frame_step(self, tmp_path, base_config):
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        config = base_config(input_file, "model", 0)
        assert config.generate_output_prefix() == "input_model_step0"

    def test_negative_frame_step(self, tmp_path, base_config):
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        config = base_config(input_file, "model", -1)
        assert config.generate_output_prefix() == "input_model_step-1"

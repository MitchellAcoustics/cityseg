import pytest
import numpy as np
import h5py
import cv2
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from PIL import Image
from cityseg.processors import VideoProcessor
from cityseg.config import Config
from cityseg.exceptions import ProcessingError


class TestVideoProcessor:
    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=Config)
        config.input = Path("test_video.mp4")
        # config.input.name = "test_video.mp4"
        config.batch_size = 2
        config.frame_step = 1
        config.disable_tqdm = True
        config.model = Mock()
        config.model.name = "test_model"
        config.get_output_path.return_value = Path("output/test_video_output.mp4")
        return config

    @pytest.fixture
    def mock_pipeline(self):
        pipeline = Mock()
        pipeline.palette = np.zeros((10, 3), dtype=np.uint8)
        pipeline.model.config.id2label = {0: "background", 1: "road"}
        return pipeline

    @pytest.fixture
    def processor(self, mock_config, mock_pipeline):
        with patch(
            "cityseg.processors.create_segmentation_pipeline"
        ) as mock_create_pipeline:
            mock_create_pipeline.return_value = mock_pipeline
            with patch("cityseg.processors.ProcessingPlan") as mock_plan:
                instance = mock_plan.return_value
                instance.plan = {
                    "process_video": True,
                    "generate_hdf": True,
                    "generate_colored_video": True,
                    "generate_overlay_video": True,
                    "analyze_results": True,
                }
                return VideoProcessor(mock_config)

    def test_init(self, processor, mock_config):
        assert processor.config == mock_config
        assert processor.pipeline is not None
        assert processor.processing_plan is not None
        assert processor.file_handler is not None
        assert processor.visualizer is not None
        assert processor.analyzer is not None

    @patch("cityseg.processors.cv2.VideoCapture")
    def test_process_video_frames(self, mock_video_capture, processor):
        # Mock video capture and its methods
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 10,
            cv2.CAP_PROP_FPS: 30.0,
        }.get(prop, 0)
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        # Mock the frame generator to return a fixed batch
        mock_batch = [
            Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(2)
        ]
        processor._frame_generator = Mock(return_value=[mock_batch])

        # Mock pipeline to return segmentation results
        processor.pipeline.return_value = [
            {"seg_map": np.zeros((480, 640), dtype=np.uint8)} for _ in range(2)
        ]

        # Mock file handler's update_hdf_file method
        processor.file_handler.update_hdf_file = Mock()

        # Call the method
        segmentation_data, metadata = processor._process_video_frames()

        # Verify results
        assert isinstance(segmentation_data, np.ndarray)
        assert len(segmentation_data) == 2
        assert "model_name" in metadata
        assert "frame_count" in metadata
        assert metadata["frame_count"] == 2
        assert processor.file_handler.update_hdf_file.called

    @patch("cityseg.processors.cv2.VideoCapture")
    def test_frame_generator(self, mock_video_capture, processor):
        # Setup mock video capture
        mock_cap = MagicMock()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]

        def side_effect():
            if frames:
                return True, frames.pop(0)
            return False, None

        mock_cap.read.side_effect = side_effect

        # Test the generator
        batches = list(processor._frame_generator(mock_cap))

        # We should have ceil(5/2) = 3 batches, with the last one containing only 1 frame
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1
        assert all(
            isinstance(frame, Image.Image) for batch in batches for frame in batch
        )

    @patch("cityseg.processors.cv2.VideoWriter")
    @patch("cityseg.processors.cv2.VideoCapture")
    def test_generate_videos(self, mock_video_capture, mock_video_writer, processor):
        # Setup mock data
        mock_segmentation = MagicMock(spec=h5py.Dataset)
        mock_segmentation.__len__.return_value = 10

        metadata = {
            "fps": 30.0,
            "frame_step": 1,
            "palette": np.zeros((10, 3), dtype=np.uint8),
        }

        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap

        # Mock segmentation data retrieval
        processor._get_video_frames_batch = Mock(
            return_value=[np.zeros((480, 640, 3), dtype=np.uint8)]
        )

        with patch(
            "cityseg.processors.get_segmentation_data_batch",
            return_value=np.zeros((1, 480, 640), dtype=np.uint8),
        ):
            # Mock visualization
            processor.visualizer.visualize_segmentation = Mock(
                return_value=[np.zeros((480, 640, 3), dtype=np.uint8)]
            )

            # Call generate_videos
            processor.generate_videos(mock_segmentation, metadata)

            # Verify that video writers were initialized and frames were written
            assert processor.visualizer.visualize_segmentation.called
            assert mock_video_writer.return_value.write.called

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("json.load")
    @patch("pathlib.Path.exists")
    def test_update_processing_history(
        self, mock_exists, mock_json_load, mock_json_dump, mock_open_func, processor
    ):
        # Setup the path to exist
        mock_exists.return_value = True

        # Mock the output path in the config
        output_path = Path("output/test_video_output.mp4")
        processor.config.get_output_path.return_value = output_path

        # Mock existing history
        empty_history = []
        mock_json_load.return_value = empty_history

        # Set up ConfigHasher to return a predictable hash
        with patch(
            "cityseg.processors.ConfigHasher.calculate_hash", return_value="test_hash"
        ):
            # Call the method
            processor._update_processing_history()

        # Verify that open was called (at least once)
        mock_open_func.assert_called()

        # Verify json.load was called to read the existing history
        mock_json_load.assert_called_once()

        # Verify json.dump was called to write the updated history
        mock_json_dump.assert_called_once()

        # Check the arguments passed to json.dump
        args, _ = mock_json_dump.call_args
        history = args[0]  # First argument to json.dump is the data

        # Verify the structure of the history entry
        assert isinstance(history, list)
        assert len(history) == 1
        assert "timestamp" in history[0]
        assert "config_hash" in history[0]
        assert history[0]["config_hash"] == "test_hash"
        assert "input_file" in history[0]
        assert "output_file" in history[0]

    @patch("cityseg.processors.VideoProcessor._process_video_frames")
    @patch("cityseg.processors.VideoProcessor.generate_videos")
    @patch("cityseg.processors.VideoProcessor._update_processing_history")
    @patch("cityseg.processors.SegmentationAnalyzer.analyze_results")
    def test_process_success(
        self,
        mock_analyze,
        mock_update_history,
        mock_generate_videos,
        mock_process_frames,
        processor,
    ):
        # Setup mocks
        mock_process_frames.return_value = (
            np.zeros((10, 480, 640), dtype=np.uint8),
            {"frame_count": 10},
        )

        # Mock the file_handler.save_hdf_file method to prevent filesystem access
        processor.file_handler.save_hdf_file = Mock()

        # Mock the file_handler.load_hdf_file method
        mock_hdf_file = Mock()
        MagicMock(spec=h5py.Dataset)
        processor.file_handler.load_hdf_file = Mock(
            return_value=(mock_hdf_file, {"frame_count": 10})
        )

        # Call the process method
        processor.process()

        # Verify that all the required steps were executed
        mock_process_frames.assert_called_once()
        mock_generate_videos.assert_called_once()
        mock_analyze.assert_called_once()
        mock_update_history.assert_called_once()

        # Verify that save_hdf_file was called
        processor.file_handler.save_hdf_file.assert_called_once()

    @patch("cityseg.processors.VideoProcessor._process_video_frames")
    def test_process_error(self, mock_process_frames, processor):
        # Setup mock to raise an exception
        mock_process_frames.side_effect = Exception("Test error")

        # Test that the process method raises ProcessingError
        with pytest.raises(ProcessingError):
            processor.process()

    @patch("cityseg.processors.cv2.VideoWriter_fourcc")
    @patch("cityseg.processors.cv2.VideoWriter")
    def test_initialize_video_writers(self, mock_video_writer, mock_fourcc, processor):
        # Setup
        mock_fourcc.return_value = "test_fourcc"
        width, height = 640, 480
        fps = 30.0

        # Call the method
        writers = processor._initialize_video_writers(width, height, fps)

        # Verify writers were created
        assert "colored" in writers
        assert "overlay" in writers
        assert mock_video_writer.call_count == 2

    @patch("cityseg.processors.cv2.VideoCapture")
    def test_get_video_frames_batch(self, mock_video_capture):
        # Setup mock
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        # Call the static method
        frames = VideoProcessor._get_video_frames_batch(mock_cap, 0, 3, 1)

        # Verify frames were retrieved
        assert len(frames) == 3
        assert all(isinstance(frame, np.ndarray) for frame in frames)
        assert mock_cap.set.call_count == 3

    @patch("cityseg.processors.cv2.VideoWriter_fourcc")
    @patch("cityseg.processors.cv2.VideoWriter")
    @patch("cityseg.processors.tqdm_context")
    def test_create_video(self, mock_tqdm, mock_video_writer, mock_fourcc, processor):
        # Setup mocks
        mock_fourcc.return_value = "fourcc_value"
        mock_video_writer_instance = Mock()
        mock_video_writer.return_value = mock_video_writer_instance

        mock_cap = MagicMock()
        # Return True for first 5 frames, then False to end the loop
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)) for _ in range(5)
        ] + [(False, None)]

        # Mock h5py.Dataset
        mock_segmentation_data = MagicMock(spec=h5py.Dataset)
        mock_segmentation_data.__getitem__.return_value = np.zeros(
            (480, 640), dtype=np.uint8
        )

        metadata = {
            "fps": 30.0,
            "width": 640,
            "height": 480,
            "frame_step": 2,
            "frame_count": 5,
            "palette": np.zeros((10, 3), dtype=np.uint8),
        }

        output_path = Path("test_output.mp4")

        # Mock tqdm context
        mock_pbar = Mock()
        mock_tqdm.__enter__.return_value = mock_pbar

        # Mock visualizer
        processor.visualizer.visualize_segmentation = Mock(
            return_value=np.zeros((480, 640, 3), dtype=np.uint8)
        )

        # Call the method
        processor._create_video(
            mock_cap, mock_segmentation_data, metadata, output_path, colored_only=True
        )

        # Verify video writer was initialized correctly
        mock_fourcc.assert_called_once_with(*"mp4v")
        mock_video_writer.assert_called_once_with(
            str(output_path), "fourcc_value", 30.0, (640, 480)
        )

        # Verify frames were read and processed
        assert mock_cap.read.call_count == 6  # 5 frames + 1 final check that fails

        # With frame_step=2, we should have accessed segmentation data for frames 0, 2, 4
        assert mock_segmentation_data.__getitem__.call_count == 3

        # Verify visualization was called for segmentation frames
        assert processor.visualizer.visualize_segmentation.call_count == 3

        # Verify frames were written to video
        assert mock_video_writer_instance.write.call_count == 5

        # Verify video writer was released
        mock_video_writer_instance.release.assert_called_once()

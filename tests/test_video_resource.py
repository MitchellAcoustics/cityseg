"""
Tests for the video resource module.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from cityseg.video_resource import VideoResource


class TestVideoResource:
    """Tests for the VideoResource class."""

    @pytest.fixture
    def mock_video_path(self):
        """Mock video path."""
        return Path("/path/to/video.mp4")

    @pytest.fixture
    def mock_cap(self):
        """Mock video capture object."""
        mock = MagicMock()
        mock.isOpened.return_value = True
        mock.get.side_effect = lambda prop: {
            0: 1920,  # width
            1: 1080,  # height
            5: 30.0,  # fps
            7: 300,   # frame count
            38: "avc1"  # codec (fake value for testing)
        }.get(prop, 0)
        
        # Mock read to return a frame and success
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock.read.return_value = (True, frame)
        
        return mock

    @patch("cv2.VideoCapture")
    def test_context_manager(self, mock_video_capture, mock_video_path, mock_cap):
        """Test the context manager interface."""
        mock_video_capture.return_value = mock_cap
        
        # Test __enter__ and __exit__
        with VideoResource(mock_video_path) as cap:
            assert cap is mock_cap
            mock_video_capture.assert_called_once_with(str(mock_video_path))
        
        # Verify that release was called
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_get_metadata(self, mock_video_capture, mock_video_path, mock_cap):
        """Test getting video metadata."""
        mock_video_capture.return_value = mock_cap
        
        resource = VideoResource(mock_video_path)
        metadata = resource.get_metadata()
        
        assert metadata["width"] == 1920
        assert metadata["height"] == 1080
        assert metadata["fps"] == 30.0
        assert metadata["frame_count"] == 300

    @patch("cv2.VideoCapture")
    @patch("cv2.cvtColor")
    def test_get_frame_batch(self, mock_cvtcolor, mock_video_capture, mock_video_path, mock_cap):
        """Test getting a batch of frames."""
        mock_video_capture.return_value = mock_cap
        
        # Mock cv2.cvtColor to return the same array
        mock_cvtcolor.side_effect = lambda frame, _: frame
        
        resource = VideoResource(mock_video_path)
        frames = resource.get_frame_batch([0, 10, 20])
        
        assert len(frames) == 3
        assert isinstance(frames[0], Image.Image)
        
        # Verify that set positions were called correctly
        assert mock_cap.set.call_count == 3
        mock_cap.set.assert_any_call(1, 0)  # First frame
        mock_cap.set.assert_any_call(1, 10)  # 10th frame
        mock_cap.set.assert_any_call(1, 20)  # 20th frame

    @patch("cv2.VideoCapture")
    @patch("cv2.cvtColor")
    def test_get_frames_by_step(self, mock_cvtcolor, mock_video_capture, mock_video_path, mock_cap):
        """Test getting frames by step size."""
        mock_video_capture.return_value = mock_cap
        
        # Make read return True for 10 frames, then False
        frame_count = [0]
        def mock_read():
            frame_count[0] += 1
            if frame_count[0] <= 10:
                return True, np.zeros((1080, 1920, 3), dtype=np.uint8)
            return False, None
        
        mock_cap.read.side_effect = mock_read
        
        # Mock cv2.cvtColor to return the same array
        mock_cvtcolor.side_effect = lambda frame, _: frame
        
        resource = VideoResource(mock_video_path)
        
        # Get every 2nd frame
        frames = resource.get_frames_by_step(frame_step=2)
        
        # We should get 5 frames (0, 2, 4, 6, 8)
        assert len(frames) == 5
        assert isinstance(frames[0], Image.Image)

    @patch("cv2.VideoCapture")
    @patch("cv2.cvtColor")
    def test_get_frame_batch_generator(self, mock_cvtcolor, mock_video_capture, mock_video_path, mock_cap):
        """Test the frame batch generator."""
        mock_video_capture.return_value = mock_cap
        
        # Mock cv2.cvtColor to return the same array
        mock_cvtcolor.side_effect = lambda frame, _: frame
        
        resource = VideoResource(mock_video_path)
        
        # Create generator
        gen = resource.get_frame_batch_generator(frame_step=10, batch_size=3)
        
        # Get first batch
        batch1 = next(gen)
        assert len(batch1) == 3
        assert isinstance(batch1[0], Image.Image)
        
        # Mock hit end of file after first batch
        mock_cap.read.return_value = (False, None)
        
        # Iteration should stop
        batches = list(gen)
        assert len(batches) == 0
        
        # Verify set positions were called correctly for the first batch
        mock_cap.set.assert_any_call(1, 0)   # First frame
        mock_cap.set.assert_any_call(1, 10)  # Second frame
        mock_cap.set.assert_any_call(1, 20)  # Third frame
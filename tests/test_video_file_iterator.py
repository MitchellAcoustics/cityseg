from pathlib import Path
from unittest.mock import patch

import pytest

from src.cityseg.video_file_iterator import VideoFileIterator


@pytest.fixture
def mock_video_files():
    # Create mock Path objects for video files
    return [Path("video1.mp4"), Path("video2.mp4")]


@pytest.fixture
def mock_directory():
    # Create a mock Path object for the directory
    return Path("/mock/directory")


def test_video_file_iterator_initializes_with_video_files(
    mock_video_files, mock_directory
):
    with patch.object(Path, "glob", return_value=mock_video_files):
        iterator = VideoFileIterator(mock_directory)
        assert iterator.video_files == mock_video_files


def test_video_file_iterator_yields_video_files(mock_video_files, mock_directory):
    with patch.object(Path, "glob", return_value=mock_video_files):
        iterator = VideoFileIterator(mock_directory)
        video_files_list = list(iterator)
        assert video_files_list == mock_video_files


def test_video_file_iterator_handles_empty_directory(mock_directory):
    with patch.object(Path, "glob", return_value=[]):
        iterator = VideoFileIterator(mock_directory)
        assert iterator.video_files == []
        assert list(iterator) == []


def test_video_file_iterator_handles_non_mp4_files(mock_directory):
    non_video_files = [Path("file1.txt"), Path("file2.doc")]
    with patch.object(Path, "glob", return_value=non_video_files):
        iterator = VideoFileIterator(mock_directory)
        assert iterator.video_files == []
        assert list(iterator) == []


def test_video_file_iterator_handles_mixed_files(mock_directory):
    mixed_files = [Path("video1.mp4"), Path("file1.txt"), Path("video2.avi")]
    with patch.object(Path, "glob", return_value=mixed_files):
        iterator = VideoFileIterator(mock_directory)
        assert iterator.video_files == [Path("video1.mp4"), Path("video2.avi")]
        assert list(iterator) == [Path("video1.mp4"), Path("video2.avi")]

from pathlib import Path
from typing import List

from loguru import logger

from .utils import get_video_files


class VideoFileIterator:
    def __init__(self, input_path: Path):
        self.input_path = input_path
        self.video_files = self._get_video_files()

    def _get_video_files(self) -> List[Path]:
        video_files = get_video_files(self.input_path)
        logger.info(f"Found {len(video_files)} video files in {self.input_path}")
        return video_files

    def __iter__(self):
        return iter(self.video_files)

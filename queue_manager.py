import queue
from typing import Optional, Any

class SingleFrameQueueManager:
    def __init__(self, maxsize: int = 1):
        self.frame_queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped_frames: int = 0
        self.processed_frames: int = 0

    def put_latest(self, item: Any) -> bool:
        if self.frame_queue.full():
            try:
                _ = self.frame_queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(item)
            return True
        except queue.Full:
            return False

    def get_latest(self) -> Optional[Any]:
        try:
            frame = self.frame_queue.get_nowait()
            self.processed_frames += 1
            return frame
        except queue.Empty:
            return None

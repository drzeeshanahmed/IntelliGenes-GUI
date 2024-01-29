# https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
import io
import os
import sys
from threading import Thread

# from threading import Thread
from PySide6.QtCore import Signal, QThread
from typing import Callable

from utils.queue import StdOut

class Worker:
    def __init__(self, process: Callable[[], None], queue: StdOut):
        super().__init__()
        self._process = process
        self._queue = queue
    
    def start(self):
        # Memory Issue while using QThread for process, seems to work with normal Thread API
        self._thread = Thread(target=self.run)
        self._thread.start()
    
    def run(self):
        self._process()
        self._queue.close()
    
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()


class CaptureOutput(QThread):
    textChanged = Signal(str)

    def __init__(self, job, stdout: StdOut):
        super().__init__()
        self._stdout = stdout
        self._job = Worker(job, stdout)
        self._text = ""

    def run(self):
        self._job.start()

        while self._job.is_alive() or not self._stdout.empty():
            line = self._stdout.read()
            if line is not None:
                self._text += line
                self.textChanged.emit(self._text)
        
        self.close()

    # Cleanup memory
    def close(self):
        self._stdout.close()
        self.quit()
        self.deleteLater()
        

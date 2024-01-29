# https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
import io
import os
import sys
from threading import Thread

# from threading import Thread
from PySide6.QtCore import Signal, QThread
from typing import Callable

class Worker:

    def __init__(self, process: Callable[[], None]):
        super().__init__()
        self._process = process
    
    def start(self):
        # Memory Issue while using QThread for process, seems to work with normal Thread API
        self._thread = Thread(target=self.run)
        self._thread.start()
    
    def run(self):
        self._process()
    
    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()


class CaptureOutput(QThread):
    textChanged = Signal(str)
    beganRunning = Signal()

    def __init__(self, job):
        super().__init__()
        self._stdout = None
        self._stdout = None
        self._ostream = None
        self._istream = None
        self._job = Worker(job)
        self._text = ""
    
    def print(self, s, end=""):
        print(s, file=self._stdout, end=end)
    
    def readAndAppendLine(self):
        try:
            line = self._istream.readline()
            if len(line) > 0:
                self._text += line
                return True
        except:
            pass
        
        return False
    
    def run(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        # creates a pair where output to w gets read from r
        r, w = os.pipe()
        self._istream, self._ostream = os.fdopen(r, "r"), os.fdopen(w, "w", 1)
        sys.stdout = self._ostream
        sys.stderr = self._ostream

        self._job.start()
        while self._job.is_alive():
            if self.readAndAppendLine():
                self.textChanged.emit(self._text)

        # Finish reading any remaining lines
        # while True:
        #     print("still reading")
        #     if self.readAndAppendLine():
        #         print("read line")
        #         self.textChanged.emit(self._text)
        #         print("next")
        #     else:
        #         break

        # Cleanup memory
        self.deleteLater()
        self._ostream.close()
        self._istream.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        
        

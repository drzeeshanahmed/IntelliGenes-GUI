# https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call

import os
import sys

# from threading import Thread
from PySide6.QtCore import Signal, QThread
from typing import Callable

class Worker(QThread):
    def __init__(self, process: Callable[[], None]):
        super().__init__()
        self._process = process
    
    def run(self):
        self._process()


class CaptureOutput(QThread):
    textChanged = Signal(str)
    jobStarted = Signal()
    jobEnded = Signal()

    def __init__(self):
        super().__init__()
        self._stdout = None
        self._stdout = None
        self._ostream = None
        self._istream = None
    
    def print(self, s, end=""):
        print(s, file=self._stdout, end=end)
    
    def run(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._closed = False
        # creates a pair where output to w gets read from r
        r, w = os.pipe()
        self._istream, self._ostream = os.fdopen(r, "r"), os.fdopen(w, "w", 1)
        sys.stdout = self._ostream
        sys.stderr = self._ostream

        _text = ""
        while not self._closed:
            try:
                # Inner while loop is necessary in order to finish reading output
                while True:
                    line = self._istream.readline()
                    self.print(line)
                    if len(line) == 0:
                        break
                    _text += line
                    self.textChanged.emit(_text)
            except:
                break
        
        self._ostream.close()
        self._istream.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def close(self):
        self._closed = True

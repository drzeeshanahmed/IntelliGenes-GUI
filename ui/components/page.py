# UI libraries
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import SignalInstance


class Page(QWidget):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ):
        super().__init__()
        self.inputFile = inputFile
        self.outputDir = outputDir
        self.onTabSelected = onTabSelected

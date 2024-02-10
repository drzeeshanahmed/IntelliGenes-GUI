import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTabWidget
from PySide6.QtCore import Signal
from pandas import DataFrame

# Custom Components
from .input.page import InputPage
from .files.page import FilesPage
from .pipeline.page import PipelinePage


class MainWindow(QMainWindow):
    # global state for input and output file
    # will be either a valid path or an empty string
    inputFile = Signal(str)
    outputDir = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntelliGenes")

        layout = QVBoxLayout()

        tabs = [
            ("Input", InputPage(inputFile=self.inputFile, outputDir=self.outputDir)),
            ("Pipeline", PipelinePage(inputFileSignal=self.inputFile, outputDirSignal=self.outputDir)),
            ("Files", FilesPage(outputDir=self.outputDir)),
        ]
        self.inputFile.emit("")
        self.outputDir.emit("")

        tab_bar = QTabWidget()
        tab_bar.setTabPosition(QTabWidget.TabPosition.North)
        tab_bar.setDocumentMode(True)

        for name, widget in tabs:
            tab_bar.addTab(widget, name)
        tab_bar.setLayout(layout)

        self.setCentralWidget(tab_bar)


def run():
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

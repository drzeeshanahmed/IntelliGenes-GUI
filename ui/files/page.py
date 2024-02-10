# UI libraries
from PySide6.QtCore import QDir, Signal, Qt, SignalInstance
from PySide6.QtWidgets import (
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QFileDialog,
)

# Custom UI libraries
from ui.components.page import Page
from ui.components.png_renderer import ImageRenderer
from ui.components.table_renderer import TableRenderer


class OutputFilesPage(Page):
    selectedFile = Signal(str)

    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)
        self.path = None

        self.rendered_widget = None
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        label = QLabel("Select an Output Directory")
        self._layout.addWidget(label)

        self.list = QListWidget()
        self._layout.addWidget(self.list)

        self.list.itemClicked.connect(
            lambda i: self.selectedFile.emit(i.data(Qt.ItemDataRole.UserRole))
        )

        self.outputDir.connect(self.setOutputPath)
        self.outputDir.connect(
            lambda text: label.setText(text if text else "Select an Output Directory")
        )
        self.onTabSelected.connect(self.updateDirectoryWidgets)
        self.selectedFile.connect(self.handleSelectedFile)

    def setOutputPath(self, path: str):
        self.path = path

    def browseDirectory(self):
        dir = QFileDialog.getExistingDirectory()
        if dir:
            self.outputDir.emit(dir)

    def updateDirectoryWidgets(self):
        self.list.clear()

        if not self.path:
            return

        dir = QDir(self.path)
        dir.setNameFilters(["*.csv", "*.png"])
        dir.setFilter(QDir.Filter.Files)
        files = dir.entryInfoList()

        for file in files:
            widget = QListWidgetItem(file.fileName())
            widget.setData(Qt.ItemDataRole.UserRole, file.absoluteFilePath())
            self.list.addItem(widget)

    def handleSelectedFile(self, path: str):
        if self.rendered_widget is not None:
            self.rendered_widget.deleteLater()

        if not path:
            self.rendered_widget = QLabel("Select a file to preview")
        elif path.endswith("png"):
            self.rendered_widget = ImageRenderer(path)
        elif path.endswith("csv"):
            self.rendered_widget = TableRenderer(path)
        else:
            self.rendered_widget = QLabel("Unsupported file type")

        self._layout.addWidget(self.rendered_widget)

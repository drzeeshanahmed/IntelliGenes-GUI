
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QFileDialog,
    QPushButton,
)
from PySide6.QtCore import QDir, Signal, Qt

from .png_renderer import ImageRenderer
from .table_renderer import TableRenderer


class FilesPage(QWidget):
    selectedDir = Signal(str)
    selectedFile = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.rendered_widget = None
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        button = QPushButton()
        button.clicked.connect(self.browseDirectory)
        self._layout.addWidget(button)

        self.list = QListWidget()
        self._layout.addWidget(self.list)

        self.list.itemClicked.connect(
            lambda i: self.selectedFile.emit(i.data(Qt.ItemDataRole.UserRole))
        )

        self.selectedDir.connect(
            lambda text: button.setText("Select Directory" if not text else text)
        )
        self.selectedDir.connect(self.updateDirectoryWidgets)
        self.selectedFile.connect(self.handleSelectedFile)
        self.selectedFile.emit("")
        self.selectedDir.emit("")

    def browseDirectory(self):
        dir = QFileDialog.getExistingDirectory()
        if dir:
            self.selectedDir.emit(dir)

    def updateDirectoryWidgets(self, path):
        self.list.clear()

        if not path:
            return

        dir = QDir(path)
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

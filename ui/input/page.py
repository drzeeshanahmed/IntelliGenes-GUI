from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QPushButton,
)
from PySide6.QtCore import SignalInstance, Qt

from ui.files.table_renderer import TableRenderer

class InputPage(QWidget):
    def __init__(
        self, inputFile: SignalInstance, outputDir: SignalInstance
    ) -> None:
        super().__init__()
        self.rendered_widget = None
        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self._layout)

        file_btn = QPushButton()
        output_btn = QPushButton()
        self._layout.addWidget(file_btn)
        self._layout.addWidget(output_btn)

        inputFile.connect(
            lambda text: file_btn.setText("Select CIGT File" if not text else text)
        )
        inputFile.connect(self.handleSelectedFile)

        outputDir.connect(
            lambda text: output_btn.setText(
                "Select Output Directory" if not text else text
            )
        )

        file_btn.clicked.connect(lambda: inputFile.emit(self.selectFile()))
        output_btn.clicked.connect(lambda: outputDir.emit(self.selectDirectory()))

    def selectFile(self):
        filename, ok = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a CSV File",
            dir="",
            filter="CSV (*.csv)",
            selectedFilter="",
        )
        return filename

    def handleSelectedFile(self, path: str):
        rendered_widget = None
        if not path:
            rendered_widget = QLabel("Select a file to preview")
        elif path.endswith("csv"):
            rendered_widget = TableRenderer(path)
        else:
            rendered_widget = QLabel("Unsupported file type")

        if self.rendered_widget is not None:
            self._layout.replaceWidget(self.rendered_widget, rendered_widget)
        else:
            self._layout.addWidget(rendered_widget)
        self.rendered_widget = rendered_widget

    def selectDirectory(self):
        dir = QFileDialog.getExistingDirectory()
        return dir

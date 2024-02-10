# UI libraries
from PySide6.QtCore import SignalInstance, Qt
from PySide6.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QPushButton,
)

# Custom libraries
from ui.components.page import Page
from ui.components.table_renderer import TableRenderer


class InputPage(Page):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)

        self.rendered_widget = None
        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self._layout)

        file_btn = QPushButton()
        output_btn = QPushButton()
        self._layout.addWidget(file_btn)
        self._layout.addWidget(output_btn)

        self.inputFile.connect(
            lambda text: file_btn.setText("Select CIGT File" if not text else text)
        )
        self.inputFile.connect(self.handleSelectedFile)

        self.outputDir.connect(
            lambda text: output_btn.setText(
                "Select Output Directory" if not text else text
            )
        )

        file_btn.clicked.connect(lambda: self.inputFile.emit(self.selectFile()))
        output_btn.clicked.connect(lambda: self.outputDir.emit(self.selectDirectory()))

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

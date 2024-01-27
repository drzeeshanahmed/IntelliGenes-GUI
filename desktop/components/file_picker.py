from PySide6.QtWidgets import QFileDialog, QPushButton
from typing import Callable
import os


class FilePicker(QPushButton):
    def __init__(self, text: str, callback: Callable[[str], None], type: str):
        super().__init__()
        self.default_text = text
        self.file = None
        self.callback = callback

        if type == "csv":
            if self.default_text is None:
                self.default_text = "Select CSV File"
            self.clicked.connect(self._select_csv)
        elif type == "png":
            if self.default_text is None:
                self.default_text = "Select PNG File"
            self.clicked.connect(self._select_png)
        elif type == "dir":
            if self.default_text is None:
                self.default_text = "Select Directory"
            self.clicked.connect(self._select_dir)
        else:
            raise ValueError("Invalid file type provided")
        
        self.setText(self.default_text)
        self.callback(None)

    def _select_csv(self):
        filename, ok = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a CSV File",
            dir="",
            filter="CSV (*.csv)",
            selectedFilter="",
        )

        if not filename:
            self.setText(self.default_text)
            self.callback(None)
        else:
            self.setText(os.path.basename(filename))
            self.callback(filename)

    def _select_dir(self) -> str:
        dir = QFileDialog.getExistingDirectory()
        
        if not dir:
            self.setText(self.default_text)
            self.callback(None)
        else:
            self.setText(os.path.basename(dir))
            self.callback(dir)


    def _select_png() -> str:
        pass

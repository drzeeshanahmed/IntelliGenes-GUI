# UI libraries
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
)

# Miscellaneous system libraries
from typing import Callable

# Custom utilities
from utils.setting import Config


class PipelineControls(QWidget):
    def __init__(
        self,
        pipelines: list[tuple[str, Config, Callable[[], None]]],
        run_button: QPushButton,
        combo_box: QComboBox,
    ) -> None:
        super().__init__()
        self.pipelines = pipelines
        self.run = run_button

        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self.widget = None
        self.update_settings(combo_box.currentIndex())
        combo_box.currentIndexChanged.connect(self.update_settings)

        self._layout.addWidget(combo_box)
        self._layout.addWidget(self.widget)
        self._layout.addWidget(run_button)

    def update_settings(self, index: int):
        _, config, _ = self.pipelines[index]
        old = self.widget
        self.widget = config.widget()
        if old is not None:
            self._layout.replaceWidget(old, self.widget)
            old.deleteLater()

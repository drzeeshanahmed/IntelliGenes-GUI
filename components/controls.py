from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
)
from PySide6.QtCore import Qt
from typing import Callable
from utils.setting import Setting


class PipelineControls(QWidget):
    def __init__(
        self,
        pipelines: list[tuple[str, list[Setting], Callable[[], None]]],
        run_button: QPushButton,
        combo_box: QComboBox,
    ) -> None:
        super().__init__()
        self.pipelines = pipelines
        self.run = run_button

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)

        self.settings_layout = QVBoxLayout()
        self.settings_layout.setSpacing(10)

        self.update_settings(combo_box.currentIndex())
        combo_box.currentIndexChanged.connect(self.update_settings)
        layout.addWidget(combo_box)

        layout.addLayout(self.settings_layout)
        layout.addWidget(run_button)

    def update_settings(self, index: int):
        _, config, _ = self.pipelines[index]
        for i in reversed(range(self.settings_layout.count())):
            self.settings_layout.itemAt(i).widget().deleteLater()

        for setting in config:
            container = QWidget()

            container_layout = QHBoxLayout()
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.addWidget(QLabel(setting.name))
            container_layout.addStretch(1)
            container.setLayout(container_layout)

            self.settings_layout.addWidget(container)
            container_layout.addWidget(setting.widget())

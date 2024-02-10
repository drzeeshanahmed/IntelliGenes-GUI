# UI Libraries
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
)

# System libraries
from typing import Any


class Setting:
    def __init__(self, name: str, value):
        self.name = name
        self.value = value

    def set(self, value):
        self.value = value

    def widget(self) -> QWidget:
        pass


class Group(Setting):
    def __init__(self, name: str, settings: list[Setting]):
        super().__init__(name, None)
        self.settings = settings

    def widget(self) -> QWidget:
        widget = QGroupBox(self.name)
        container_layout = QVBoxLayout()
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        container_layout.setSpacing(3)
        container_layout.setContentsMargins(10, 5, 10, 5)
        widget.setLayout(container_layout)

        for setting in self.settings:
            container_layout.addWidget(setting.widget())

        return widget


class Config:
    def __init__(self, settings: list[Setting]):
        self.settings = settings

    def get(self, name: str) -> Any | None:
        for setting in self.settings:
            if setting.name == name:
                return setting.name
            elif isinstance(setting, Group):
                for s in setting.settings:
                    if s.name == name:
                        return s.value
        return None

    def widget(self) -> QWidget:
        widget = QWidget()
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)

        for setting in self.settings:
            settings_layout.addWidget(setting.widget())

        settings_layout.setSpacing(30)
        widget.setLayout(settings_layout)
        return widget


class IntSetting(Setting):
    def __init__(self, name: str, value: int, min: int, max: int, step: int = 1):
        super().__init__(name, value)
        self.min = min
        self.max = max
        self.step = step

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        sb = QSpinBox()
        sb.setValue(self.value)
        sb.setMinimumWidth(100)
        sb.setMinimum(self.min)
        sb.setMaximum(self.max)
        sb.setSingleStep(self.step)
        sb.valueChanged.connect(self.set)

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(sb)

        return widget


class FloatSetting(Setting):
    def __init__(
        self, name: str, value: float, min: float, max: float, step: int = 0.05
    ):
        super().__init__(name, value)
        self.min = min
        self.max = max
        self.step = step

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        sb = QDoubleSpinBox()
        sb.setValue(self.value)
        sb.setMinimumWidth(100)
        sb.setMinimum(self.min)
        sb.setMaximum(self.max)
        sb.setSingleStep(self.step)
        sb.valueChanged.connect(self.set)

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(sb)

        return widget


class BoolSetting(Setting):
    def __init__(self, name: str, value: bool):
        super().__init__(name, value)

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        cb = QCheckBox()
        cb.setChecked(self.value)
        cb.stateChanged.connect(lambda s: self.set(s == Qt.CheckState.Checked.value))

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(cb)

        return widget


class StrSetting(Setting):
    def __init__(self, name: str, value):
        super().__init__(name, value)

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        le = QLineEdit("value")
        le.textChanged.connect(self.set)

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(le)

        return widget


class StrChoiceSetting(Setting):
    def __init__(self, name: str, value: int, options: list[str]):
        super().__init__(name, value)
        self.options = options

        if value not in options:
            raise ValueError("Value not in provided options")

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        cb = QComboBox()
        cb.addItems(self.options)
        cb.setCurrentText(self.value)
        cb.currentTextChanged.connect(self.set)

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(cb)

        return widget

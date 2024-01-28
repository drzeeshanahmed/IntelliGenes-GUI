import os
from PySide6.QtWidgets import (
    QWidget,
    QFileDialog,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QCheckBox,
    QPushButton,
)
from PySide6.QtCore import Qt


class Setting:
    # String can be 'csv', 'int', 'float' 'bool', or 'str'
    def __init__(self, name: str, type: str, value):
        self.name = name
        self.type = type
        self.value = value

    def set(self, value):
        self.value = value

    def widget(self) -> QWidget:
        pass


class IntSetting(Setting):
    def __init__(self, name: str, value: int, min: int, max: int, step: int = 1):
        self.min = min
        self.max = max
        self.step = step

        super().__init__(name, "int", value)

    def widget(self):
        widget = QSpinBox()
        widget.setValue(self.value)
        widget.setMinimumWidth(100)
        widget.setMinimum(self.min)
        widget.setMaximum(self.max)
        widget.setSingleStep(self.step)

        widget.valueChanged.connect(self.set)
        return widget


class FloatSetting(Setting):
    def __init__(
        self, name: str, value: float, min: float, max: float, step: int = 0.05
    ):
        self.min = min
        self.max = max
        self.step = step
        super().__init__(name, "float", value)

    def widget(self):
        widget = QDoubleSpinBox()
        widget.setValue(self.value)
        widget.setMinimumWidth(100)
        widget.setMinimum(self.min)
        widget.setMaximum(self.max)
        widget.setSingleStep(self.step)

        widget.valueChanged.connect(self.set)
        return widget


class BoolSetting(Setting):
    def __init__(self, name: str, value: bool):
        super().__init__(name, "bool", value)

    def widget(self):
        widget = QCheckBox()
        widget.setChecked(self.value)

        widget.stateChanged.connect(
            lambda state: self.set(state == Qt.CheckState.Checked.value)
        )
        return widget


class StrSetting(Setting):
    def __init__(self, name: str, value):
        super().__init__(name, "str", value)

    def widget(self):
        widget = QLineEdit("value")
        widget.textChanged.connect(self.set)
        return widget


class CSVSetting(Setting):
    def __init__(self, name: str, path):
        super().__init__(name, "csv", path)

    def _getText(self):
        # self.value is either empty or None
        return (
            os.path.basename(self.value)
            if self.value is not None
            else "Select CSV File"
        )

    def _getFile(self, widget: QPushButton):
        filename, ok = QFileDialog.getOpenFileName(
            parent=widget,
            caption="Select a CSV File",
            dir="",
            filter="CSV (*.csv)",
            selectedFilter="",
        )

        self.set(filename if filename else None)
        widget.setText(self._getText())

    def widget(self):
        widget = QPushButton(self._getText())
        widget.clicked.connect(lambda: self._getFile(widget))
        return widget


class DirectorySetting(Setting):
    def __init__(self, name: str, path):
        super().__init__(name, "dir", path)

    def _getText(self):
        # self.value is either empty or None
        return (
            os.path.basename(self.value)
            if self.value is not None
            else "Select Directory"
        )

    def _getDir(self, widget: QPushButton):
        dir = QFileDialog.getExistingDirectory()

        self.set(dir if dir else None)
        widget.setText(self._getText())

    def widget(self):
        widget = QPushButton(self._getText())
        widget.clicked.connect(lambda: self._getDir(widget))
        return widget


class StrChoiceSetting(Setting):
    def __init__(self, name: str, value: int, options: list[str]):
        self.options = options

        if value not in options:
            raise ValueError("Value not in provided options")

        super().__init__(name, "str_dropdown", value)

    def widget(self):
        widget = QComboBox()
        widget.addItems(self.options)
        widget.setCurrentText(self.value)
        widget.currentTextChanged.connect(self.set)
        return widget

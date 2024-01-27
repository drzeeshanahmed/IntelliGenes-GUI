from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QCheckBox,
)
from PySide6.QtCore import Qt

from components.file_picker import FilePicker


class Setting:
    # String can be 'csv', 'int', 'float' 'bool', or 'str'
    def __init__(self, name: str, type: str, default_value):
        self.name = name
        if type not in ["csv", "int", "float", "bool", "str", "dir"]:
            raise ValueError("Invalid type")
        self.type = type
        self.value = default_value

    def set(self, value):
        self.value = value


class PipelineConfiguration(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)

        self.pipelines: list[tuple[str, list[Setting]]] = [
            (
                "Feature Selection",
                [
                    Setting("Input", "csv", None),
                    Setting("Output", "dir", None),
                    Setting("Random State", "int", 42),
                    Setting("Test Size", "float", 0.3),
                    Setting("Normalize", "bool", False),
                    Setting("Recursive Feature Elimination", "bool", True),
                    Setting("Pearson's Correlation", "bool", True),
                    Setting("Chi-Squared Test", "bool", True),
                    Setting("Analysis of Variance", "bool", True),
                ],
            ),
            (
                "Feature Classification",
                [
                    Setting("Input", "csv", None),
                    Setting("Selected Features", "dir", None),
                    Setting("Random State", "int", 42),
                    Setting("Test Size", "float", 0.3),
                    Setting("Normalize", "bool", False),
                    Setting("Recursive Feature Elimination", "bool", True),
                    Setting("Pearson's Correlation", "bool", True),
                    Setting("Chi-Squared Test", "bool", True),
                    Setting("Analysis of Variance", "bool", True),
                ],
            ),
            ("Selection and Classification", []),
        ]

        self.settings_layout = QVBoxLayout()
        self.settings_layout.setSpacing(10)

        combobox = QComboBox()
        for name, _ in self.pipelines:
            combobox.addItem(name)
        combobox.currentIndexChanged.connect(self.update_settings)
        self.update_settings(0)

        layout.addWidget(QLabel("Pipeline:"))
        layout.addWidget(combobox)

        layout.addWidget(QLabel("Settings:"))
        layout.addLayout(self.settings_layout)

    def update_settings(self, index: int):
        _, config = self.pipelines[index]
        for i in reversed(range(self.settings_layout.count())):
            self.settings_layout.itemAt(i).widget().deleteLater()
        
        for setting in config:
            container = QWidget()
            container.setContentsMargins(0, 0, 0, 0)

            container_layout = QHBoxLayout()
            container_layout.addWidget(QLabel(setting.name))
            container_layout.addStretch(0.5)
            container.setLayout(container_layout)
            container.setContentsMargins(0, 0, 0, 0)

            self.settings_layout.addWidget(container)

            match setting.type:
                case "str":
                    widget = QLineEdit()
                    # Important to assign s=setting so that the current element is captured
                    # Closures in python are evaluated when called and so not setting this would result in
                    # the final setting (last one) being called
                    widget.textChanged.connect(lambda t, s=setting: s.set(t))
                    widget.setText(setting.value)
                case "int":
                    widget = QSpinBox()
                    widget.setMinimumWidth(100)
                    widget.valueChanged.connect(lambda i, s=setting: s.set(i))
                    widget.setValue(setting.value)
                case "float":
                    widget = QDoubleSpinBox()
                    widget.setMinimumWidth(100)
                    widget.valueChanged.connect(lambda f, s=setting: s.set(f))
                    widget.setValue(setting.value)
                    widget.setMinimum(0)
                    widget.setMaximum(1)
                    widget.setSingleStep(0.05)
                case "bool":
                    widget = QCheckBox()
                    widget.stateChanged.connect(
                        lambda state, s=setting: s.set(state == Qt.CheckState.Checked.value)
                    )
                    widget.setChecked(setting.value)
                case "csv":
                    widget = FilePicker(setting.value, setting.set, "csv")
                case "dir":
                    widget = FilePicker(setting.value, setting.set, "dir")

            container_layout.addWidget(widget)

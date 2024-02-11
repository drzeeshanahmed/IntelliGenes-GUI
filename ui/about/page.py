# UI Libraries
from PySide6.QtCore import Qt, SignalInstance
from PySide6.QtWidgets import QVBoxLayout, QLabel

# Custom UI libraries
from ui.components.page import Page


class AboutPage(Page):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)
        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self._layout)

        self._layout.addWidget(QLabel("About IntelliGenes Desktop"))
        self._layout.addWidget(
            QLabel(
                """
To learn more, you can visit the GitHub page (https://github.com/drzeeshanahmed/IntelliGenes_Desktop).
"""
            )
        )

from PySide6.QtWidgets import QWidget, QHBoxLayout

from components.controls import PipelineConfiguration

class PipelinePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        controls = PipelineConfiguration()
        layout.addWidget(controls)
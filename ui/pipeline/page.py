from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox
from PySide6.QtCore import SignalInstance
from typing import Callable

from ui.pipeline.controls import PipelineControls
from ui.pipeline.console import PipelineConsole
from utils.output_capture import CaptureOutput

from utils.intelligenes_pipelines import (
    classification_pipeline,
    feature_selection_pipeline,
    select_and_classify_pipeline,
)
from utils.setting import Setting

job = None
output = None

class PipelinePage(QWidget):
    def __init__(self, changeDirSignal: SignalInstance) -> None:
        super().__init__()
        pipelines: list[tuple[str, list[Setting], Callable[[], None]]] = [
            feature_selection_pipeline(changeDirSignal),
            classification_pipeline(changeDirSignal),
            select_and_classify_pipeline(changeDirSignal),
        ]

        layout = QHBoxLayout()
        self.setLayout(layout)

        combo_box = QComboBox()
        run_button = QPushButton("Run")

        for name, inputs, callback in pipelines:
            combo_box.addItem(name)
        combo_box.setCurrentIndex(2)

        console = PipelineConsole()
        
        # Run process in a separate thread and capture output for the console
        def run():
            # need to declare global so that variables doesn't immediately go out of scope
            global job, output
            output = CaptureOutput(pipelines[combo_box.currentIndex()][2])
            output.textChanged.connect(console.setText)
            output.started.connect(lambda: run_button.setDisabled(True))
            output.finished.connect(lambda: run_button.setDisabled(False))
            output.start()
        
        run_button.clicked.connect(run)
        # combo_box.currentIndexChanged.connect(console.clear)

        controls = PipelineControls(pipelines, run_button, combo_box)

        layout.addWidget(controls)
        layout.setStretch(0, 1)
        layout.addWidget(console)
        layout.setStretch(1, 2)


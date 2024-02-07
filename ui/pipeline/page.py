from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox
from PySide6.QtCore import SignalInstance
from typing import Callable

from .controls import PipelineControls
from .console import PipelineConsole
from utils.capture_output import CaptureOutput
from utils.stdout import StdOut
from utils.setting import Setting

from intelligenes.intelligenes_pipelines import (
    select_and_classify_pipeline,
    classification_pipeline,
    feature_selection_pipeline,
)

class PipelinePage(QWidget):
    def __init__(self, changeDirSignal: SignalInstance) -> None:
        super().__init__()

        self.stdout = StdOut()
        self.output = CaptureOutput(self.stdout)

        pipelines: list[tuple[str, list[Setting], Callable[[], None]]] = [
            select_and_classify_pipeline(changeDirSignal, self.stdout),
            feature_selection_pipeline(changeDirSignal, self.stdout),
            classification_pipeline(changeDirSignal, self.stdout),
        ]

        layout = QVBoxLayout()
        self.setLayout(layout)

        combo_box = QComboBox()
        run_button = QPushButton("Run")

        for name, inputs, callback in pipelines:
            combo_box.addItem(name)
        combo_box.setCurrentIndex(0)

        console = PipelineConsole()
        
        self.output.textChanged.connect(console.setText)
        self.output.started.connect(lambda: run_button.setDisabled(True))
        self.output.finished.connect(lambda: run_button.setDisabled(False))

        # Run process in a separate thread and capture output for the console
        def run():
            self.stdout.open()
            self.output.load_job(pipelines[combo_box.currentIndex()][2])
            self.output.start() # closes stdout when finished (need to reopen)
        
        run_button.clicked.connect(run)
        # combo_box.currentIndexChanged.connect(console.clear)

        controls = PipelineControls(pipelines, run_button, combo_box)

        layout.addWidget(controls)
        layout.setStretch(0, 1)
        layout.addWidget(console)
        layout.setStretch(1, 2)


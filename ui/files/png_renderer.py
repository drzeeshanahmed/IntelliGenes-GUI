from PySide6.QtWidgets import QLabel, QScrollArea
from PySide6.QtGui import QPixmap


class ImageRenderer(QScrollArea):
    def __init__(self, path: str):
        super().__init__()
        label = QLabel()
        pixmap = QPixmap(path) # .scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(pixmap)
        self.setWidget(label)
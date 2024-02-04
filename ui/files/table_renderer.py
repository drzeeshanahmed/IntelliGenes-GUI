import pandas as pd
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


class TableRenderer(QTableWidget):
    def __init__(self, path: str):
        super().__init__()
        df = pd.read_csv(path)

        self.setRowCount(df.shape[0])  # header
        self.setColumnCount(df.shape[1])
        # To prevent eliding of data
        self.setWordWrap(False)

        self.setHorizontalHeaderLabels(df.columns)

        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                widget = QTableWidgetItem(str(df.iloc[r, c]))
                self.setItem(r, c, widget)

from PyQt5.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MplWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.fig=Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.navi_toolbar)

        # self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

    def getFigure(self):
        return self.fig

    def draw(self):
        self.canvas.draw()

    def clear(self):
        self.fig.clf()
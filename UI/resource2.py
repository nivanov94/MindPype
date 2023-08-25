"""
This file somewhat builds on the previous tutorial, but also enables the saving and loading of a session in the GUI.

It does this by adding a save and load button to the GUI, which calls the save_session and load_session methods in 
the session class. The save_session method saves the session to a pickle file, and the load_session method loads 
the session from a pickle file.
"""
from enum import Enum
from functools import partial
import typing
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QGraphicsPolygonItem



class AnnotationScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super(AnnotationScene, self).__init__(parent)
        self.image_item = QtWidgets.QGraphicsPixmapItem()
        self.image_item.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.addItem(self.image_item)
    
    def do(self):
        polygon_item = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF([QtCore.QPointF(0,0), QtCore.QPointF(100,100), QtCore.QPointF(200,200)]))
        self.addItem(polygon_item)
        return polygon_item
    
class AnnotationView(QtWidgets.QGraphicsView):
    factor = 2.0

    def __init__(self, parent=None):
        super(AnnotationView, self).__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)


# Describes the big window
class AnnotationWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(AnnotationWindow, self).__init__(parent)
        self.m_view = AnnotationView()
        self.m_scene = AnnotationScene(self)
        self.m_view.setScene(self.m_scene)

        self.setCentralWidget(self.m_view)
        self.create_menus()

    def create_menus(self):

        menu_instructions = self.menuBar().addMenu("try")
        polygon_action = menu_instructions.addAction("Polygon")
        polygon_action.triggered.connect(self.m_scene.do)

        menu_save = self.menuBar().addMenu("save")
        save_action = menu_save.addAction("save")
        save_action.triggered.connect(self.save)

        menu_load  = self.menuBar().addMenu("load")
        load_action = menu_load.addAction("load")
        load_action.triggered.connect(self.load)

    def save(self):
        name = QtWidgets.QFileDialog.getSaveFileName(self, app.tr('Save File'), "", "INI Files (*.ini)")    
        output_file = QtCore.QFile(name[0])
        output_file.open(QtCore.QIODevice.WriteOnly)
        stream_out = QtCore.QDataStream(output_file)
        stream_out << app.allWidgets()
        output_file.close()

    def load(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, app.tr('Open File'), "", "INI Files (*.ini)")
        input_file = QtCore.QFile(name[0])
        input_file.open(QtCore.QIODevice.ReadOnly)
        stream_in = QtCore.QDataStream(input_file)
        stream_in >> app.allWidgets()
        input_file.close()

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = AnnotationWindow()
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())
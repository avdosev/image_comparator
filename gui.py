from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtGui import QPixmap


class ImageViewer(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)

        label = QLabel(self)
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)

        self.resize(pixmap.width(), pixmap.height())  # fit window to the image
        self.setWindowTitle('PyQt5 Image Viewer')


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    if len(sys.argv) < 2:
        sys.exit(__doc__)
    app = QApplication(sys.argv)
    image_viewer = ImageViewer(sys.argv[1])
    image_viewer.show()
    sys.exit(app.exec_())

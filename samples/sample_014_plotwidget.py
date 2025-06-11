import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QFont
import pyqtgraph as pg
import numpy as np


class MultiAxisChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Multi-Axis Chart (Single PlotWidget, Linked Y-axes)")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --------------------------------------------------
        # Main PlotWidget
        # --------------------------------------------------
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # --------------------------------------------------
        # PlotItem 1 (Main PlotItem) - 左側のY軸 (Value 1)
        # --------------------------------------------------
        main_plot_item = self.plot_widget.plotItem

        main_plot_item.setLabel('left', 'Value 1', units='units')
        main_plot_item.setLabel('bottom', 'Time')  # X軸のラベルはここに一つだけ
        main_plot_item.setTitle('Combined Trend Chart')

        # X軸の範囲を固定
        main_plot_item.setXRange(0, 10)

        # メインデータ (Value 1)
        x1 = np.linspace(0, 10, 100)
        y1 = np.sin(x1 * 2) + np.random.rand(100) * 0.5
        main_plot_item.plot(x1, y1, pen='b', name='Trend 1')

        # ティックラベルのフォントサイズ設定
        xaxis = main_plot_item.getAxis('bottom')
        font = QFont()
        font.setPointSize(12)
        xaxis.setTickFont(font)

        yaxis1 = main_plot_item.getAxis('left')
        font1_y = QFont()
        font1_y.setPointSize(10)
        yaxis1.setTickFont(font1_y)

        # --------------------------------------------------
        # PlotItem 2 - 右側のY軸 (Value 2)
        # --------------------------------------------------
        self.p2 = pg.ViewBox()
        main_plot_item.scene().addItem(self.p2)

        # 新しいViewBoxのX軸をメインのViewBoxのX軸にリンクさせる
        self.p2.setXLink(main_plot_item.vb)

        # 右側に新しいAxisItemを追加
        self.axis2 = pg.AxisItem('right')
        main_plot_item.layout.addItem(self.axis2, 2, 3)  # row=2 (中央), col=3 (右端) に配置
        self.axis2.setLabel('Value 2', units='units')

        # AxisItemとViewBoxをリンクさせる
        self.axis2.linkToView(self.p2)

        # セカンダリデータ (Value 2) - 桁数を大きくしてみる
        x2 = np.linspace(0, 10, 100)
        y2 = np.cos(x2 * 3) * 100 + np.random.rand(100) * 30
        self.plot_item2 = pg.PlotCurveItem(x=x2, y=y2, pen='r', name='Trend 2')
        self.p2.addItem(self.plot_item2)

        # ティックラベルのフォントサイズ設定
        font2_y = QFont()
        font2_y.setPointSize(10)
        self.axis2.setTickFont(font2_y)

        # --------------------------------------------------
        # ウィンドウサイズ変更時のViewBoxの自動調整
        # --------------------------------------------------
        # main_plot_item.layout は GraphicsLayout のインスタンス
        main_plot_item.layout.setColumnStretchFactor(1, 1)  # プロットエリア（中央の列）の伸縮設定

        # **** ここが修正点 ****
        # PlotItemのViewBoxのsigResizedシグナルを接続
        main_plot_item.vb.sigResized.connect(self.updateViews)
        # 初回表示時にも調整
        self.updateViews()

    def updateViews(self):
        # メインのViewBoxのサイズが変更されたときに、セカンダリのViewBoxのサイズも合わせる
        # p2.setGeometry()でp2の表示領域をmain_plot_item.vbの表示領域に合わせる
        self.p2.setGeometry(self.plot_widget.plotItem.vb.sceneBoundingRect())
        # linkedViewChanged は ViewBox間のリンクに使用されるため、
        # ここではX軸が既にリンクされているため、特定の引数は不要、あるいは不要な場合がある。
        # 今回は単にsetGeometryで十分なため、コメントアウトまたは削除。
        # self.p2.linkedViewChanged(self.plot_widget.plotItem.vb, self.p2.XAxis)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MultiAxisChartWindow()
    window.show()
    sys.exit(app.exec())
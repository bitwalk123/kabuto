import math
import sys
import random

import numpy as np
import pyqtgraph as pg
from scipy.interpolate import make_smoothing_spline
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot, QSharedMemory

# ロギング設定
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 共有メモリのキー定義 ---
SHARED_MEMORY_KEY_X = "my_shared_array_x_key"
SHARED_MEMORY_KEY_Y = "my_shared_array_y_key"
DTYPE = np.float64
INT_DTYPE = np.int64
ITEM_SIZE = np.dtype(DTYPE).itemsize  # float64のバイトサイズ
INT_ITEM_SIZE = np.dtype(INT_DTYPE).itemsize  # int64のバイトサイズ


class DataGeneratorWorker(QObject):
    notifySmoothLineReady = Signal()
    notifyNewData = Signal(int, float)
    # 内部的な共有メモリ準備完了通知
    _internalSharedMemoryReady = Signal()

    def __init__(self, max_data: int, parent: QObject = None):
        super().__init__(parent)  # parentはNoneとして渡される
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_data = max_data

        self.shm_x = QSharedMemory(SHARED_MEMORY_KEY_X)
        self.shm_y = QSharedMemory(SHARED_MEMORY_KEY_Y)

        self.shm_size_x = max_data * INT_ITEM_SIZE
        self.shm_size_y = max_data * ITEM_SIZE

        self.shm_initialized = False

        self.shm_data_x = None
        self.shm_data_y = None

    @Slot()
    def initialize_shared_memory(self):
        """
        ワーカースレッド内で共有メモリの作成と初期化を行うスロット。
        このスロットは、QThreadが起動した後に呼ばれる。
        """
        if self.shm_initialized:
            return

        # --- QSharedMemory の作成とアタッチ ---
        # create()を試み、もし既に存在すればattach()する
        if not self.shm_x.create(self.shm_size_x):
            if self.shm_x.error() == QSharedMemory.SharedMemoryError.AlreadyExists:
                self.logger.warning(f"Shared memory for X already exists, attaching: {self.shm_x.errorString()}")
                if not self.shm_x.attach():
                    self.logger.error(f"Failed to attach shared memory for X: {self.shm_x.errorString()}")
                    raise RuntimeError("Failed to attach shared memory for X")
            else:
                self.logger.error(f"Failed to create shared memory for X: {self.shm_x.errorString()}")
                raise RuntimeError("Failed to create shared memory for X")
        else:
            self.logger.info(
                f"Shared memory for X created with key '{SHARED_MEMORY_KEY_X}' and size {self.shm_size_x} bytes.")

        if not self.shm_y.create(self.shm_size_y):
            if self.shm_y.error() == QSharedMemory.SharedMemoryError.AlreadyExists:
                self.logger.warning(f"Shared memory for Y already exists, attaching: {self.shm_y.errorString()}")
                if not self.shm_y.attach():
                    self.logger.error(f"Failed to attach shared memory for Y: {self.shm_y.errorString()}")
                    raise RuntimeError("Failed to attach shared memory for Y")
            else:
                self.logger.error(f"Failed to create shared memory for Y: {self.shm_y.errorString()}")
                raise RuntimeError("Failed to create shared memory for Y")
        else:
            self.logger.info(
                f"Shared memory for Y created with key '{SHARED_MEMORY_KEY_Y}' and size {self.shm_size_y} bytes.")

        # 共有メモリのバイト配列ビューをNumPy配列としてマップ
        self.shm_data_x = np.ndarray(
            shape=(self.max_data,),
            dtype=INT_DTYPE,
            buffer=self.shm_x.data()
        )
        self.shm_data_y = np.ndarray(
            shape=(self.max_data,),
            dtype=DTYPE,
            buffer=self.shm_y.data()
        )

        # 初期化（共有メモリの内容が不定なので0でクリア）
        self.shm_data_x.fill(0)
        self.shm_data_y.fill(0)

        self.shm_initialized = True
        # 共有メモリの準備ができたことを親スレッドに通知
        self._internalSharedMemoryReady.emit()

    @Slot(int)
    def generateNewData(self, counter: int):
        """
        メインスレッドからのリクエストに応じて新しいデータを生成し、
        共有メモリに書き込むスロット。
        """
        if not self.shm_initialized:
            self.logger.warning("Shared memory not initialized yet, skipping data generation.")
            return

        x = counter
        y = math.sin(x / 10.) + random.random() + 1

        # 個別データ点はこれまで通りシグナルで送る
        self.notifyNewData.emit(x, y)

        # ロックの前に共有メモリが有効か確認する（念のため）
        if not (self.shm_x.isAttached() and self.shm_y.isAttached()):
            self.logger.error("Shared memory not attached in worker, cannot lock.")
            return

        # 共有メモリをロックし、データを書き込む
        if not self.shm_x.lock():
            self.logger.error(f"Failed to lock shared memory for X (worker): {self.shm_x.errorString()}")
            return
        if not self.shm_y.lock():
            self.logger.error(f"Failed to lock shared memory for Y (worker): {self.shm_y.errorString()}")
            self.shm_x.unlock()  # 片方がロック失敗したらもう片方もアンロック
            return

        try:
            self.shm_data_x[counter] = x
            self.shm_data_y[counter] = y

            # スムージングには最低5点必要
            if counter >= 5:
                # 共有メモリ上のデータが更新されたことをメインスレッドに通知
                self.notifySmoothLineReady.emit()

        finally:
            self.shm_x.unlock()  # アンロック
            self.shm_y.unlock()  # アンロック

    def __del__(self):
        """
        DataGeneratorWorkerオブジェクトが削除される際に共有メモリをデタッチする。
        """
        if self.shm_x.isAttached():
            self.logger.info(f"Detaching shared memory for X: {SHARED_MEMORY_KEY_X}")
            self.shm_x.detach()
        if self.shm_y.isAttached():
            self.logger.info(f"Detaching shared memory for Y: {SHARED_MEMORY_KEY_Y}")
            self.shm_y.detach()


class ThreadDataGenerator(QThread):
    requestNewData = Signal(int)  # メインスレッドからのデータ生成リクエスト
    threadReady = Signal()  # スレッドがイベントループを開始したことを通知
    sharedMemoryReady = Signal()  # ワーカーが共有メモリの準備を完了したことを通知

    def __init__(self, max_data: int, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        # DataGeneratorWorkerの親をNoneにしてmoveToThreadが有効になるようにする
        self.worker = DataGeneratorWorker(max_data)

        # ワーカーオブジェクトをこのQThreadのイベントループに移動
        self.worker.moveToThread(self)

        # シグナルとスロットの接続
        self.started.connect(self.thread_ready)  # QThread起動時にthreadReadyシグナルを発行
        self.requestNewData.connect(self.worker.generateNewData)  # メインスレッドからのリクエストをワーカーのスロットに接続

        # ワーカーの内部シグナルを、このスレッドの公開シグナルに接続
        self.worker._internalSharedMemoryReady.connect(self.sharedMemoryReady)

        # QThreadが起動したら、ワーカーの共有メモリ初期化メソッドを実行
        self.started.connect(self.worker.initialize_shared_memory)

    @Slot()
    def thread_ready(self):
        self.threadReady.emit()

    def run(self):
        """
        QThreadの実行エントリポイント。
        このメソッドが完了するとスレッドは終了する。exec()でイベントループを開始させる。
        """
        self.logger.info(f"{self.__class__.__name__} for data generation: run() method started. Entering event loop...")
        self.exec()  # このスレッドのイベントループを開始
        self.logger.info(f"{self.__class__.__name__} for data generation: run() method finished. Event loop exited.")


class TrendGraph(pg.PlotWidget):
    def __init__(self, max_data: int):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_data = max_data

        self.shm_x = QSharedMemory(SHARED_MEMORY_KEY_X)
        self.shm_y = QSharedMemory(SHARED_MEMORY_KEY_Y)

        self.shm_size_x = max_data * INT_ITEM_SIZE
        self.shm_size_y = max_data * ITEM_SIZE

        self.shm_data_x = None
        self.shm_data_y = None

        self.x_data_points = []
        self.y_data_points = []

        self.showGrid(x=True, y=True, alpha=0.5)
        self.setXRange(0, max_data)

        self.data_points_item = pg.ScatterPlotItem(
            size=5,
            pen=pg.mkPen(color=(0, 255, 255), width=1),
            brush=pg.mkBrush(color=(0, 255, 255)),
            symbol='o',
            pxMode=True,
            antialias=False
        )
        self.addItem(self.data_points_item)

        self.smoothed_line_item = pg.PlotDataItem(
            pen=pg.mkPen(color=(255, 255, 0), width=1),
            pxMode=True,
            antialias=False
        )
        self.addItem(self.smoothed_line_item)

    @Slot()
    def initialize_shared_memory(self):
        """
        メインスレッド内で共有メモリへのアタッチを行うスロット。
        ワーカーが共有メモリを作成した後に呼ばれる。
        """
        # 既にアタッチ済みなら何もしない
        if self.shm_x.isAttached() and self.shm_y.isAttached():
            return

            # 共有メモリへのアタッチ
        if not self.shm_x.attach():
            self.logger.error(f"Failed to attach shared memory for X (graph): {self.shm_x.errorString()}")
            raise RuntimeError("Failed to attach shared memory for X (graph)")
        else:
            self.logger.info(f"Shared memory for X attached to graph with key '{SHARED_MEMORY_KEY_X}'.")

        if not self.shm_y.attach():
            self.logger.error(f"Failed to attach shared memory for Y (graph): {self.shm_y.errorString()}")
            self.shm_x.detach()  # 片方失敗したらもう片方もデタッチ
            raise RuntimeError("Failed to attach shared memory for Y (graph)")
        else:
            self.logger.info(f"Shared memory for Y attached to graph with key '{SHARED_MEMORY_KEY_Y}'.")

        # 共有メモリのバイト配列ビューをNumPy配列としてマップ
        self.shm_data_x = np.ndarray(
            shape=(self.max_data,),
            dtype=INT_DTYPE,
            buffer=self.shm_x.data(),
        )
        self.shm_data_y = np.ndarray(
            shape=(self.max_data,),
            dtype=DTYPE,
            buffer=self.shm_y.data(),
        )
        self.logger.info("TrendGraph: Shared memory views initialized.")

    @Slot(int, float)
    def addPoints(self, x: int, y: float):
        """
        個々のデータ点を受け取り、グラフに追加するスロット。
        """
        self.x_data_points.append(x)
        self.y_data_points.append(y)

        # 必要に応じて、表示範囲外の古いデータを削除
        if len(self.x_data_points) > self.max_data:
            self.x_data_points.pop(0)
            self.y_data_points.pop(0)

        self.data_points_item.setData(self.x_data_points, self.y_data_points)

        self.logger.info(f"追加データ: X={x}, Y={y}")

    @Slot()
    def updateSmoothedLine(self):
        """
        共有メモリから最新のデータを読み込み、スムージングラインを更新するスロット。
        """
        # 共有メモリがまだアタッチされていなければ処理をスキップ
        if not (self.shm_x.isAttached() and self.shm_y.isAttached()):
            self.logger.warning("TrendGraph: Shared memory not attached yet, skipping smooth line update.")
            return

        # 共有メモリをロックし、データを読み込む
        if not self.shm_x.lock():
            self.logger.error(f"Failed to lock shared memory for X (graph): {self.shm_x.errorString()}")
            return
        if not self.shm_y.lock():
            self.logger.error(f"Failed to lock shared memory for Y (graph): {self.shm_y.errorString()}")
            self.shm_x.unlock()  # 片方がロック失敗したらもう片方もアンロック
            return

        try:
            # 現在データがどこまで入っているかを知るために、メインウィンドウのカウンタを使う
            # ★注意: この方法はExampleへの依存性が高い。より堅牢な設計では、シグナルで必要な情報を渡す。
            current_data_count = QApplication.instance()._main_window_instance.count
            if current_data_count == 0 or current_data_count < 5:  # スムージングの最低点数
                return

            # 共有メモリ上のNumPy配列ビューからデータをスライスして取得
            xs_data = self.shm_data_x[0:current_data_count]
            ys_data = self.shm_data_y[0:current_data_count]

            # メインスレッドでスムージング処理を実行
            spl = make_smoothing_spline(xs_data, ys_data)
            ys_smoothed = spl(xs_data)

            self.smoothed_line_item.setData(xs_data, ys_smoothed)
            self.logger.info(f"スムージングライン更新: データ点数={len(xs_data)}")

        finally:
            self.shm_x.unlock()  # アンロック
            self.shm_y.unlock()  # アンロック

    def __del__(self):
        """
        TrendGraphオブジェクトが削除される際に共有メモリからデタッチする。
        """
        if self.shm_x.isAttached():
            self.logger.info(f"Detaching shared memory for X: {SHARED_MEMORY_KEY_X}")
            self.shm_x.detach()
        if self.shm_y.isAttached():
            self.logger.info(f"Detaching shared memory for Y: {SHARED_MEMORY_KEY_Y}")
            self.shm_y.detach()


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setWindowTitle("リアルタイム風トレンドグラフ (Shared Memory)")
        self.setFixedSize(800, 600)

        self.max_data = 180
        self.count = 0

        # メインウィンドウのインスタンスをQApplication経由で保存（TrendGraphからの参照用）
        # ★注意: この方法は簡易的。より堅牢な設計では、シグナルで必要な情報を渡す。
        QApplication.instance()._main_window_instance = self

        self.chart = None  # TrendGraphのインスタンスは共有メモリ準備後に作成

        self.data_generator_thread = ThreadDataGenerator(self.max_data)
        self.data_generator_thread.threadReady.connect(self.on_data_generator_thread_ready)

        # DataGeneratorThreadから共有メモリ準備完了シグナルを受け取る
        self.data_generator_thread.sharedMemoryReady.connect(self.on_shared_memory_ready)

        # ワーカースレッドを開始
        self.data_generator_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)
        # ★ここを修正しました★ タイマーを繰り返し発火させるため False に設定
        self.timer.setSingleShot(False)

    @Slot()
    def on_shared_memory_ready(self):
        """
        ワーカースレッドが共有メモリの作成・初期化を完了した後に呼ばれるスロット。
        ここでTrendGraphのインスタンスを作成し、共有メモリにアタッチする。
        """
        self.logger.info("Main thread: Shared memory is ready from worker.")
        # TrendGraphのインスタンス作成と共有メモリのアタッチ処理
        self.chart = TrendGraph(self.max_data)
        self.setCentralWidget(self.chart)
        self.chart.initialize_shared_memory()  # ここでTrendGraphが共有メモリにアタッチする

        # シグナル接続
        self.data_generator_thread.worker.notifyNewData.connect(self.chart.addPoints)
        self.data_generator_thread.worker.notifySmoothLineReady.connect(self.chart.updateSmoothedLine)

        # 全ての準備が整ったので、タイマーを開始
        self.timer.start(1000)  # 1000msごとにデータ生成をトリガー

    def closeEvent(self, event: QCloseEvent):
        """
        アプリケーション終了時のクリーンアップ処理。
        """
        # タイマー停止
        if self.timer.isActive():
            self.timer.stop()

        # スレッドの安全な終了
        if self.data_generator_thread.isRunning():
            self.logger.info("Stopping data generator thread...")
            # QThreadのquit()はイベントループを終了させる
            self.data_generator_thread.quit()
            # スレッドが終了するまで待機
            self.data_generator_thread.wait()
            self.logger.info("The data generator thread safely terminated.")

        # chart が None の場合があるのでチェック
        if self.chart:
            # chart の __del__ が呼ばれることを保証する（明示的なデタッチは__del__に任せる）
            pass

        super().closeEvent(event)
        event.accept()

    @Slot()
    def on_data_generator_thread_ready(self):
        """
        データ生成スレッドがそのイベントループを開始したときに呼ばれるスロット。
        """
        self.logger.info("Data generator thread started its event loop.")

    @Slot()
    def update_chart(self):
        """
        QTimerから定期的に呼び出され、データ生成をトリガーするスロット。
        """
        if self.count >= self.max_data:
            self.timer.stop()
            self.logger.info("リアルタイム更新が終了しました。")
            return

        # ワーカーにデータ生成をリクエスト
        self.data_generator_thread.requestNewData.emit(self.count)
        self.count += 1


if __name__ == "__main__":
    # pyqtgraphのグローバル設定
    pg.setConfigOption('background', 'k')  # 黒背景 (ダークモード風)
    pg.setConfigOption('foreground', 'w')  # 白前景 (テキストなど)

    app = QApplication(sys.argv)
    window = Example()
    window.show()
    sys.exit(app.exec())
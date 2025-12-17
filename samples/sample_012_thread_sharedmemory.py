import sys
import numpy as np
import pyqtgraph as pg
import math
import random
import logging

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PySide6.QtCore import QThread, QObject, Signal, Slot, QSharedMemory, QTimer
from scipy.interpolate import make_smoothing_spline, splev

# ロギング設定
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 共有メモリのキー定義 ---
SHARED_MEMORY_KEY_X = "my_shared_array_x_key_reboot" # キー名を変更して以前のキーと混同しないように
SHARED_MEMORY_KEY_Y = "my_shared_array_y_key_reboot"
SHARED_MEMORY_KEY_Y_SMOOTHED = "my_shared_array_y_smoothed_key_reboot" # スムージング済みY座標用

DTYPE = np.float64
INT_DTYPE = np.int64
ITEM_SIZE = np.dtype(DTYPE).itemsize # float64のバイトサイズ
INT_ITEM_SIZE = np.dtype(INT_DTYPE).itemsize # int64のバイトサイズ

# --- スムージング関数 ---
# スムージングスプラインを生成し、その評価を行うヘルパー関数
# 本来のやり方（make_smoothing_spline(x, y)）を踏襲
def create_bspline_and_smoothed_data(x, y):
    # データ点数が5点未満の場合は None を返す
    if len(x) < 5:
        logger.debug(f"DataGeneratorWorker: スムージングに必要なデータ点数に未到達 ({len(x)}点)")
        return None, None

    try:
        # あなたの元のやり方: make_smoothing_spline(x, y) で lam は指定しない
        tck = make_smoothing_spline(x, y)

        # splev を使用してスムージングスプラインを評価
        # x_smoothed の生成は、元のサンプルと同様に、データ範囲で500点を生成
        x_smoothed = np.linspace(x.min(), x.max(), 500)
        y_smoothed = splev(x_smoothed, tck)

        logger.debug(f"DataGeneratorWorker: スムージングデータ生成。データ点数={len(x)}")
        return x_smoothed, y_smoothed
    except Exception as e:
        logger.error(f"Error creating spline: {e}")
        return None, None  # エラー時も None を返す


# DataGeneratorWorker クラスの残りの部分は、前回のコードと同じです。
# update_data メソッド内で create_bspline_and_smoothed_data(self.x_data, self.y_data) を呼び出す形になります。
# 5点未満の場合に shm_np_y_smoothed を np.nan で埋める処理も維持されます。

# DataGeneratorWorker クラスの残りの部分は、前回のコードと同じです。
# update_data メソッド内で create_bspline_and_smoothed_data(self.x_data, self.y_data) を呼び出す形になります。
# 5点未満の場合に shm_np_y_smoothed を np.nan で埋める処理も維持されます。
# --- データ生成ワーカー ---
class DataGeneratorWorker(QObject):
    notifyNewData = Signal(int, float) # 個別の新しいデータ点を通知 (散布図用)
    notifySmoothLineReady = Signal(int)   # スムージングデータが共有メモリに書き込まれたことを通知 (データ点数を追加)
    _internalSharedMemoryReady = Signal() # 内部的に共有メモリが初期化されたことをExampleに通知

    def __init__(self, max_data: int, parent: QObject = None):
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_data = max_data

        self.shm_x = QSharedMemory(SHARED_MEMORY_KEY_X)
        self.shm_y = QSharedMemory(SHARED_MEMORY_KEY_Y)
        self.shm_y_smoothed = QSharedMemory(SHARED_MEMORY_KEY_Y_SMOOTHED)

        self.shm_size_x = max_data * INT_ITEM_SIZE
        self.shm_size_y = max_data * ITEM_SIZE
        self.shm_size_y_smoothed = max_data * ITEM_SIZE

        self.shm_initialized = False

        self.shm_data_x = None
        self.shm_data_y = None
        self.shm_data_y_smoothed = None

    @Slot()
    def initialize_shared_memory(self):
        """
        ワーカースレッド内で共有メモリの作成と初期化を行うスロット。
        """
        if self.shm_initialized:
            return

        # X座標の共有メモリの作成とアタッチ
        if not self.shm_x.create(self.shm_size_x):
            if self.shm_x.error() == QSharedMemory.AlreadyExists:
                self.logger.warning(f"Shared memory for X already exists, attaching: {self.shm_x.errorString()}")
                if not self.shm_x.attach():
                    self.logger.error(f"Failed to attach shared memory for X: {self.shm_x.errorString()}")
                    raise RuntimeError("Failed to attach shared memory for X")
            else:
                self.logger.error(f"Failed to create shared memory for X: {self.shm_x.errorString()}")
                raise RuntimeError("Failed to create shared memory for X")
        else:
            self.logger.info(f"Shared memory for X created with key '{SHARED_MEMORY_KEY_X}' and size {self.shm_size_x} bytes.")

        # Y座標の共有メモリの作成とアタッチ
        if not self.shm_y.create(self.shm_size_y):
            if self.shm_y.error() == QSharedMemory.AlreadyExists:
                self.logger.warning(f"Shared memory for Y already exists, attaching: {self.shm_y.errorString()}")
                if not self.shm_y.attach():
                    self.logger.error(f"Failed to attach shared memory for Y: {self.shm_y.errorString()}")
                    raise RuntimeError("Failed to attach shared memory for Y")
            else:
                self.logger.error(f"Failed to create shared memory for Y: {self.shm_y.errorString()}")
                raise RuntimeError("Failed to create shared memory for Y")
        else:
            self.logger.info(f"Shared memory for Y created with key '{SHARED_MEMORY_KEY_Y}' and size {self.shm_size_y} bytes.")

        # スムージング済みY座標の共有メモリの作成とアタッチ
        if not self.shm_y_smoothed.create(self.shm_size_y_smoothed):
            if self.shm_y_smoothed.error() == QSharedMemory.AlreadyExists:
                self.logger.warning(f"Shared memory for Y_SMOOTHED already exists, attaching: {self.shm_y_smoothed.errorString()}")
                if not self.shm_y_smoothed.attach():
                    self.logger.error(f"Failed to attach shared memory for Y_SMOOTHED: {self.shm_y_smoothed.errorString()}")
                    raise RuntimeError("Failed to attach shared memory for Y_SMOOTHED")
            else:
                self.logger.error(f"Failed to create shared memory for Y_SMOOTHED: {self.shm_y_smoothed.errorString()}")
                raise RuntimeError("Failed to create shared memory for Y_SMOOTHED")
        else:
            self.logger.info(f"Shared memory for Y_SMOOTHED created with key '{SHARED_MEMORY_KEY_Y_SMOOTHED}' and size {self.shm_size_y_smoothed} bytes.")

        # NumPy配列ビューの初期化
        self.shm_data_x = np.ndarray(shape=(self.max_data,), dtype=INT_DTYPE, buffer=self.shm_x.data())
        self.shm_data_y = np.ndarray(shape=(self.max_data,), dtype=DTYPE, buffer=self.shm_y.data())
        self.shm_data_y_smoothed = np.ndarray(shape=(self.max_data,), dtype=DTYPE, buffer=self.shm_y_smoothed.data())

        # 共有メモリの初期化（ゼロ埋め）
        self.shm_data_x.fill(0)
        self.shm_data_y.fill(0)
        self.shm_data_y_smoothed.fill(0)

        self.shm_initialized = True
        self.logger.info("DataGeneratorWorker: Shared memory initialized.")
        self._internalSharedMemoryReady.emit() # Exampleクラスに共有メモリ準備完了を通知

    @Slot(int)
    def generateNewData(self, counter: int):
        """
        新しいデータを生成し、共有メモリに書き込み、スムージングも行うスロット。
        """
        if not self.shm_initialized:
            self.logger.warning("DataGeneratorWorker: Shared memory not initialized yet, skipping list_item generation.")
            return

        x = counter
        y = math.sin(x / 10.) + random.random() + 1 # 乱数を加えることで波形にノイズを付加

        self.notifyNewData.emit(x, y) # 生のデータ点はこれまで通りシグナルで送る (散布図用)

        # 共有メモリへのアクセスをロックする
        # ロック順序をX, Y, Y_SMOOTHEDで統一
        if not self.shm_x.lock():
            self.logger.error(f"Failed to lock SHM_X: {self.shm_x.errorString()}"); return
        if not self.shm_y.lock():
            self.logger.error(f"Failed to lock SHM_Y: {self.shm_y.errorString()}"); self.shm_x.unlock(); return
        if not self.shm_y_smoothed.lock():
            self.logger.error(f"Failed to lock SHM_Y_SMOOTHED: {self.shm_y_smoothed.errorString()}"); self.shm_x.unlock(); self.shm_y.unlock(); return

        try:
            # 共有メモリに生のデータを書き込む
            self.shm_data_x[counter] = x
            self.shm_data_y[counter] = y

            # スムージング処理をDataGeneratorWorker内で実行
            current_data_count = counter + 1 # 現在のデータ点数

            if current_data_count >= 5: # スムージングには最低5点必要
                # 共有メモリから現在の全データを読み込む (ロック中なので安全)
                # 注: ここでビュー全体をコピーしないことでオーバーヘッドを避ける
                xs_data_for_smoothing = self.shm_data_x[0:current_data_count]
                ys_data_for_smoothing = self.shm_data_y[0:current_data_count]

                # スムージングを実行
                spl = make_smoothing_spline(xs_data_for_smoothing, ys_data_for_smoothing)
                ys_smoothed = spl(xs_data_for_smoothing)

                # スムージング結果を専用の共有メモリに書き込む
                self.shm_data_y_smoothed[0:current_data_count] = ys_smoothed

                self.logger.debug(f"DataGeneratorWorker: スムージングデータ生成。データ点数={len(ys_smoothed)}")
                self.notifySmoothLineReady.emit(current_data_count) # スムージング済みデータが利用可能になったことをUIに通知
            else:
                 self.logger.debug(f"DataGeneratorWorker: スムージングに必要なデータ点数に未到達 ({current_data_count}点)")

        finally:
            # ロック解除は逆順で
            self.shm_y_smoothed.unlock()
            self.shm_y.unlock()
            self.shm_x.unlock()

    def __del__(self):
        """
        DataGeneratorWorkerオブジェクトが削除される際に共有メモリをデタッチする。
        """
        self.logger.info("DataGeneratorWorker: __del__ called.")
        try:
            if self.shm_x.isAttached():
                self.logger.info(f"Detaching shared memory for X: {SHARED_MEMORY_KEY_X}")
                self.shm_x.detach()
            if self.shm_y.isAttached():
                self.logger.info(f"Detaching shared memory for Y: {SHARED_MEMORY_KEY_Y}")
                self.shm_y.detach()
            if self.shm_y_smoothed.isAttached():
                self.logger.info(f"Detaching shared memory for Y_SMOOTHED: {SHARED_MEMORY_KEY_Y_SMOOTHED}")
                self.shm_y_smoothed.detach()
        except RuntimeError as e:
            self.logger.warning(f"DataGeneratorWorker __del__ error: {e}. Shared memory might already be deleted by OS.")


class ThreadDataGenerator(QThread):
    requestNewData = Signal(int)
    # threadReady = Signal() # 廃止

    def __init__(self, max_data: int, parent=None):
        super().__init__(parent)
        self.worker = DataGeneratorWorker(max_data)
        self.worker.moveToThread(self)

        self.started.connect(self.worker.initialize_shared_memory) # スレッド開始時に共有メモリ初期化
        # self.started.connect(self.thread_ready) # 廃止
        self.requestNewData.connect(self.worker.generateNewData)

    # def thread_ready(self): # 廃止
    #     self.threadReady.emit()

    def run(self):
        self.exec()  # イベントループを開始


# --- グラフ描画ウィジェット ---
class TrendGraph(pg.PlotWidget):
    def __init__(self, max_data: int):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_data = max_data

        # 共有メモリのキーを再度指定
        self.shm_x = QSharedMemory(SHARED_MEMORY_KEY_X)
        self.shm_y = QSharedMemory(SHARED_MEMORY_KEY_Y)
        self.shm_y_smoothed = QSharedMemory(SHARED_MEMORY_KEY_Y_SMOOTHED)

        self.shm_size_x = max_data * INT_ITEM_SIZE
        self.shm_size_y = max_data * ITEM_SIZE
        self.shm_size_y_smoothed = max_data * ITEM_SIZE

        self.shm_data_x = None
        self.shm_data_y = None
        self.shm_data_y_smoothed = None

        # グラフアイテムの初期化
        self.plot_item = self.getPlotItem()
        self.plot_item.setTitle("Real-time Data Trend")
        self.plot_item.setLabel('left', 'Value')
        self.plot_item.setLabel('bottom', 'Time')
        self.plot_item.setXRange(0, max_data) # X軸の範囲を固定

        # 生のデータポイント用プロットアイテム (青い点)
        self.scatter_plot_item = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 255), pxMode=True)
        self.plot_item.addItem(self.scatter_plot_item)

        # スムージングライン用プロットアイテム (黄色の線)
        self.smoothed_line_item = pg.PlotDataItem(pen=pg.mkPen('y', width=2))
        self.plot_item.addItem(self.smoothed_line_item)

        # 散布図の点を保持するリスト（シグナルで個別に受け取るため）
        self.scatter_x_data = []
        self.scatter_y_data = []

    @Slot()
    def initialize_shared_memory(self):
        """
        メインスレッド内で共有メモリへのアタッチを行うスロット。
        """
        # X座標の共有メモリへのアタッチ
        if not self.shm_x.attach():
            self.logger.error(f"Failed to attach shared memory for X (graph): {self.shm_x.errorString()}")
            raise RuntimeError("Failed to attach shared memory for X (graph)")
        else:
            self.logger.info(f"Shared memory for X attached to graph with key '{SHARED_MEMORY_KEY_X}'.")

        # Y座標の共有メモリへのアタッチ
        if not self.shm_y.attach():
            self.logger.error(f"Failed to attach shared memory for Y (graph): {self.shm_y.errorString()}")
            self.shm_x.detach() # 失敗したらXもデタッチ
            raise RuntimeError("Failed to attach shared memory for Y (graph)")
        else:
            self.logger.info(f"Shared memory for Y attached to graph with key '{SHARED_MEMORY_KEY_Y}'.")

        # スムージング済みY座標の共有メモリへのアタッチ
        if not self.shm_y_smoothed.attach():
            self.logger.error(f"Failed to attach shared memory for Y_SMOOTHED (graph): {self.shm_y_smoothed.errorString()}")
            self.shm_x.detach(); self.shm_y.detach() # 失敗したらX,Yもデタッチ
            raise RuntimeError("Failed to attach shared memory for Y_SMOOTHED (graph)")
        else:
            self.logger.info(f"Shared memory for Y_SMOOTHED attached to graph with key '{SHARED_MEMORY_KEY_Y_SMOOTHED}'.")

        # NumPy配列ビューの初期化
        self.shm_data_x = np.ndarray(shape=(self.max_data,), dtype=INT_DTYPE, buffer=self.shm_x.data())
        self.shm_data_y = np.ndarray(shape=(self.max_data,), dtype=DTYPE, buffer=self.shm_y.data())
        self.shm_data_y_smoothed = np.ndarray(shape=(self.max_data,), dtype=DTYPE, buffer=self.shm_y_smoothed.data())
        self.logger.info("TrendGraph: Shared memory views initialized.")

    @Slot(int, float)
    def addPoints(self, x: int, y: float):
        """
        シグナルから受け取った単一のデータ点を散布図プロットに追加するスロット。
        """
        self.scatter_x_data.append(x)
        self.scatter_y_data.append(y)
        self.scatter_plot_item.setData(self.scatter_x_data, self.scatter_y_data)
        self.logger.debug(f"TrendGraph: Added point ({x}, {y}) to scatter plot. Current points: {len(self.scatter_x_data)}")


    @Slot(int)
    def updateSmoothedLine(self, actual_data_count: int):
        """
        共有メモリから最新のスムージング済みデータを読み込み、スムージングラインを更新するスロット。
        このスロットはDataGeneratorWorkerからnotifySmoothLineReadyシグナルで呼び出される。
        """
        # 共有メモリがまだアタッチされていなければ処理をスキップ
        if not (self.shm_x.isAttached() and self.shm_y_smoothed.isAttached()):
            self.logger.warning("TrendGraph: Shared memory not fully attached yet, skipping smooth line update.")
            return

        # 共有メモリを全てロックし、データを読み込む
        # ロック順序をX, Y, Y_SMOOTHEDで統一
        if not self.shm_x.lock(): self.logger.error(f"Failed to lock SHM_X (graph): {self.shm_x.errorString()}"); return
        if not self.shm_y.lock(): # Yも参照しないがロック順序を保つためロック
            self.logger.error(f"Failed to lock SHM_Y (graph): {self.shm_y.errorString()}"); self.shm_x.unlock(); return
        if not self.shm_y_smoothed.lock():
            self.logger.error(f"Failed to lock SHM_Y_SMOOTHED (graph): {self.shm_y_smoothed.errorString()}"); self.shm_x.unlock(); self.shm_y.unlock(); return

        try:
            self.logger.debug(f"TrendGraph: updateSmoothedLine called. actual_data_count = {actual_data_count}")

            if actual_data_count == 0 or actual_data_count < 5:
                self.logger.debug(f"TrendGraph: Not enough list_item for smoothing. actual_data_count = {actual_data_count}. Clearing smooth line.")
                self.smoothed_line_item.setData([], []) # データが少ない場合はラインをクリアしておく
                return

            # Xデータとスムージング済みYデータを共有メモリから読み込む
            # NumPyのスライス操作はビューを返すため、余分なコピーは発生しない
            xs_data = self.shm_data_x[0:actual_data_count]
            ys_smoothed_data = self.shm_data_y_smoothed[0:actual_data_count]

            # スムージング済みのデータを直接プロット
            self.smoothed_line_item.setData(xs_data, ys_smoothed_data)
            self.logger.debug(f"スムージングライン更新 (Worker生成): データ点数={len(xs_data)}")

        finally:
            # ロック解除は逆順で
            self.shm_y_smoothed.unlock()
            self.shm_y.unlock()
            self.shm_x.unlock()

    def __del__(self):
        """
        TrendGraphオブジェクトが削除される際に共有メモリからデタッチする。
        """
        self.logger.info("TrendGraph: __del__ called.")
        try:
            if self.shm_x.isAttached():
                self.logger.info(f"Detaching shared memory for X: {SHARED_MEMORY_KEY_X}")
                self.shm_x.detach()
            if self.shm_y.isAttached():
                self.logger.info(f"Detaching shared memory for Y: {SHARED_MEMORY_KEY_Y}")
                self.shm_y.detach()
            if self.shm_y_smoothed.isAttached():
                self.logger.info(f"Detaching shared memory for Y_SMOOTHED: {SHARED_MEMORY_KEY_Y_SMOOTHED}")
                self.shm_y_smoothed.detach()
        except RuntimeError as e:
            self.logger.warning(f"TrendGraph __del__ error: {e}. Shared memory might already be deleted by OS.")


# --- メインウィンドウ ---
class Example(QMainWindow):
    # シグナルの定義
    requestNewData = Signal(int)

    def __init__(self, max_data: int = 100):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setWindowTitle("PySide6 Multi-threaded GUI with Shared Memory (Smoothed in Worker)")
        self.setGeometry(100, 100, 800, 600)

        self.max_data = max_data
        self.count = 0 # データ点数カウンター

        # UI要素の作成
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.graph = TrendGraph(self.max_data)
        self.layout.addWidget(self.graph)

        self.status_label = QPushButton("Waiting for Data Generation...")
        self.status_label.setEnabled(False) # 通常は無効
        self.layout.addWidget(self.status_label)

        # QThreadのセットアップ
        self.data_generator_thread = ThreadDataGenerator(self.max_data)
        # スレッド開始時にワーカーの共有メモリ初期化スロットを呼び出す (ThreadDataGenerator内で接続済み)

        # ワーカーが共有メモリの初期化を完了したら、メインスレッド側のグラフもアタッチし、タイマーを開始する
        self.data_generator_thread.worker._internalSharedMemoryReady.connect(self.graph.initialize_shared_memory)
        self.data_generator_thread.worker._internalSharedMemoryReady.connect(self.start_data_generation_auto)

        # データの生成リクエストと結果のシグナル/スロット接続
        self.requestNewData.connect(self.data_generator_thread.requestNewData)
        self.data_generator_thread.worker.notifyNewData.connect(self.graph.addPoints) # 生のデータ点を直接プロット (散布図用)
        # スムージングデータがワーカーで準備できたらグラフの更新を要求
        self.data_generator_thread.worker.notifySmoothLineReady.connect(self.graph.updateSmoothedLine)

        self.data_generator_thread.start()
        self.logger.info("Data generator thread started its event loop.")

        # メインスレッドのタイマー設定 (ここではまだスタートしない)
        self.timer = QTimer(self)
        self.timer.setInterval(100) # 100msごとにデータ更新をリクエスト
        self.timer.timeout.connect(self.update_chart)

    # ★追加: 自動開始用のスロット
    @Slot()
    def start_data_generation_auto(self):
        self.timer.start()
        self.status_label.setText("Generating Data...")
        self.logger.info("Data generation started automatically after shared memory initialization.")


    def update_chart(self):
        """
        チャートを更新する。新しいデータ生成をワーカーにリクエストする。
        """
        if self.count < self.max_data:
            self.requestNewData.emit(self.count) # ワーカーにデータ生成をリクエスト
            self.count += 1
        else:
            self.timer.stop()
            self.status_label.setText("Data Generation Finished")
            self.logger.info("Data generation finished.")

    def closeEvent(self, event):
        """
        アプリケーション終了時にスレッドを終了し、共有メモリを解放する。
        """
        self.timer.stop()
        if self.data_generator_thread.isRunning():
            self.logger.info("Terminating list_item generator thread...")
            self.data_generator_thread.quit()
            self.data_generator_thread.wait(5000) # 最大5秒待機
            if self.data_generator_thread.isRunning():
                self.logger.warning("Data generator thread did not terminate cleanly. Terminating forcefully.")
                self.data_generator_thread.terminate()

        super().closeEvent(event)
        self.logger.info("Application closed.")

if __name__ == "__main__":
    # 既存の共有メモリを確実にクリーンアップするため、アプリケーション開始前にデタッチを試みる
    # 新しいキー名でクリーンアップ
    temp_shm_x = QSharedMemory(SHARED_MEMORY_KEY_X)
    temp_shm_y = QSharedMemory(SHARED_MEMORY_KEY_Y)
    temp_shm_y_smoothed = QSharedMemory(SHARED_MEMORY_KEY_Y_SMOOTHED)

    if temp_shm_x.attach():
        temp_shm_x.detach()
        logger.info(f"Cleaned up previous shared memory: {SHARED_MEMORY_KEY_X}")
    if temp_shm_y.attach():
        temp_shm_y.detach()
        logger.info(f"Cleaned up previous shared memory: {SHARED_MEMORY_KEY_Y}")
    if temp_shm_y_smoothed.attach():
        temp_shm_y_smoothed.detach()
        logger.info(f"Cleaned up previous shared memory: {SHARED_MEMORY_KEY_Y_SMOOTHED}")

    pg.setConfigOption('background', 'k')  # 黒背景 (ダークモード風)
    pg.setConfigOption('foreground', 'w')  # 白前景 (テキストなど)

    app = QApplication(sys.argv)
    window = Example(max_data=180) # max_data を Example コンストラクタに渡す
    window.show()
    sys.exit(app.exec())
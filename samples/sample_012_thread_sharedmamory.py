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
# このキーは、共有メモリを作成する側とアタッチする側で同じである必要がある
SHARED_MEMORY_KEY_X = "my_shared_array_x_key"
SHARED_MEMORY_KEY_Y = "my_shared_array_y_key"
# NumPy配列のデータ型とサイズを定義 (バイトサイズ計算用)
# ここではfloat64を想定
DTYPE = np.float64
INT_DTYPE = np.int64  # x_datapoints用
ITEM_SIZE = np.dtype(DTYPE).itemsize  # float64のバイトサイズ
INT_ITEM_SIZE = np.dtype(INT_DTYPE).itemsize  # int64のバイトサイズ


class DataGeneratorWorker(QObject):
    # シグナルにデータ自体は渡さない（共有メモリ上のデータ更新を通知するだけ）
    # 引数なし、または更新範囲などのメタデータのみ
    notifySmoothLineReady = Signal()
    notifyNewData = Signal(int, float)  # 個別データ点はこれまで通り

    def __init__(self, max_data: int, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_data = max_data

        # --- QSharedMemory の設定 ---
        self.shm_x = QSharedMemory(SHARED_MEMORY_KEY_X)
        self.shm_y = QSharedMemory(SHARED_MEMORY_KEY_Y)

        # 共有メモリのサイズを計算 (最大データ数 * 各要素のバイトサイズ)
        self.shm_size_x = max_data * INT_ITEM_SIZE
        self.shm_size_y = max_data * ITEM_SIZE

        # 共有メモリの作成を試みる（存在しなければ作成）
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

        # 共有メモリのバイト配列ビュー
        # NumPy配列として直接操作できるようにする
        self.shm_data_x = np.ndarray(
            shape=(max_data,),
            dtype=INT_DTYPE,
            buffer=self.shm_x.data()
        )
        self.shm_data_y = np.ndarray(
            shape=(max_data,),
            dtype=DTYPE,
            buffer=self.shm_y.data()
        )

        # 初期化（共有メモリの内容が不定なので0でクリア）
        self.shm_data_x.fill(0)
        self.shm_data_y.fill(0)

    @Slot(int)
    def generateNewData(self, counter: int):
        """
        サンプルデータ生成とスムージング、共有メモリへの書き込み
        """
        x = counter
        y = math.sin(x / 10.) + random.random() + 1

        # 個別データ点はこれまで通りシグナルで送る (オーバーヘッドは小さい)
        self.notifyNewData.emit(x, y)

        # --- 共有メモリへの書き込み ---
        if not self.shm_x.lock():  # ロック
            self.logger.error(f"Failed to lock shared memory for X (worker): {self.shm_x.errorString()}")
            return
        if not self.shm_y.lock():  # ロック
            self.logger.error(f"Failed to lock shared memory for Y (worker): {self.shm_y.errorString()}")
            self.shm_x.unlock()  # ロックが一つでも失敗したら、もう片方もアンロック
            return

        try:
            # NumPy配列のデータを直接共有メモリに書き込む（メモリビュー経由）
            self.shm_data_x[counter] = x
            self.shm_data_y[counter] = y

            # スムージングには最低5点必要
            if counter >= 5:
                # 注: make_smoothing_splineはNumPy配列をコピーして渡す必要があるため、
                # ここでは共有メモリの直接操作ではなく、NumPy配列スライスを渡します。
                # ただし、スライスはビューなので、データ自体はコピーされません。
                # splは新しいNumPy配列を返します。
                spl = make_smoothing_spline(
                    self.shm_data_x[0:counter + 1],  # 共有メモリ上のデータからビューを作成して渡す
                    self.shm_data_y[0:counter + 1]
                )
                # スプライン結果は新しい配列なので、これを共有メモリに書き戻す場合は
                # その領域を確保するか、別の共有メモリにする必要があります。
                # 今回はsmoothing_splineの結果を再度書き戻すのではなく、
                # メインスレッドが共有メモリの生データを読み込み、
                # メインスレッド側でspline計算するように変更します。

                # ★重要変更★ workerは生データを共有メモリに書き込むだけに集中
                # スムージング処理はメインスレッドに移管し、グラフの描画スロットで実行
                # これにより、共有メモリの役割が「生データの共有」に限定され、
                # 各スレッドの役割分担が明確になります。
                self.notifySmoothLineReady.emit()  # データが更新されたことを通知

        finally:
            self.shm_x.unlock()  # アンロック
            self.shm_y.unlock()  # アンロック

    def __del__(self):
        # アプリケーション終了時に共有メモリをデタッチ
        # create()で作成した側がdetach()すると、共有メモリはOSから解放される
        # attach()した側はdetach()しても、他のプロセス/スレッドが使用していれば残る
        # プロデューサーであるworkerがdetach()を呼ぶのが適切
        if self.shm_x.isAttached():
            self.logger.info(f"Detaching shared memory for X: {SHARED_MEMORY_KEY_X}")
            self.shm_x.detach()
        if self.shm_y.isAttached():
            self.logger.info(f"Detaching shared memory for Y: {SHARED_MEMORY_KEY_Y}")
            self.shm_y.detach()


class ThreadDataGenerator(QThread):
    requestNewData = Signal(int)
    threadReady = Signal()

    def __init__(self, max_data: int, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.worker = DataGeneratorWorker(max_data)
        self.worker.moveToThread(self)

        self.started.connect(self.thread_ready)
        self.requestNewData.connect(self.worker.generateNewData)

    @Slot()  # Slotデコレータを追加
    def thread_ready(self):
        self.threadReady.emit()

    def run(self):
        self.logger.info(f"{self.__class__.__name__} for data generation: run() method started. Entering event loop...")
        self.exec()  # イベントループを開始
        self.logger.info(f"{self.__class__.__name__} for data generation: run() method finished. Event loop exited.")


class TrendGraph(pg.PlotWidget):
    def __init__(self, max_data: int):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_data = max_data

        # --- QSharedMemory の設定 (Main Thread 側) ---
        self.shm_x = QSharedMemory(SHARED_MEMORY_KEY_X)
        self.shm_y = QSharedMemory(SHARED_MEMORY_KEY_Y)

        # 共有メモリのサイズを計算 (念のため、作成時と同じ計算)
        self.shm_size_x = max_data * INT_ITEM_SIZE
        self.shm_size_y = max_data * ITEM_SIZE

        # 共有メモリへのアタッチを試みる（ワーカーが作成済みであることを期待）
        if not self.shm_x.attach():
            self.logger.error(f"Failed to attach shared memory for X (graph): {self.shm_x.errorString()}")
            raise RuntimeError("Failed to attach shared memory for X (graph)")
        else:
            self.logger.info(f"Shared memory for X attached to graph with key '{SHARED_MEMORY_KEY_X}'.")

        if not self.shm_y.attach():
            self.logger.error(f"Failed to attach shared memory for Y (graph): {self.shm_y.errorString()}")
            raise RuntimeError("Failed to attach shared memory for Y (graph)")
        else:
            self.logger.info(f"Shared memory for Y attached to graph with key '{SHARED_MEMORY_KEY_Y}'.")

        # 共有メモリのバイト配列ビュー
        self.shm_data_x = np.ndarray(
            shape=(max_data,),
            dtype=INT_DTYPE,
            buffer=self.shm_x.data(),
            # 読み取り専用として安全に扱う
            # flags = 'C_CONTIGUOUS,OWNDATA' などでコピーを強制することも可能だが、目的から外れる
        )
        self.shm_data_y = np.ndarray(
            shape=(max_data,),
            dtype=DTYPE,
            buffer=self.shm_y.data(),
        )

        # データを保持するリスト（個々のデータ点用）
        self.x_data_points = []
        self.y_data_points = []

        self.showGrid(x=True, y=True, alpha=0.5)
        self.setXRange(0, max_data)

        # データ点
        self.data_points_item = pg.ScatterPlotItem(  # 変数名を変更
            size=5,
            pen=pg.mkPen(color=(0, 255, 255), width=1),
            brush=pg.mkBrush(color=(0, 255, 255)),
            symbol='o',
            pxMode=True,
            antialias=False
        )
        self.addItem(self.data_points_item)

        self.smoothed_line_item = pg.PlotDataItem(  # 変数名を変更
            pen=pg.mkPen(color=(255, 255, 0), width=1),
            pxMode=True,
            antialias=False
        )
        self.addItem(self.smoothed_line_item)

    @Slot(int, float)  # Slotデコレータを追加
    def addPoints(self, x: int, y: float):
        # データをリストに追加（個々のデータ点用）
        self.x_data_points.append(x)
        self.y_data_points.append(y)

        # 必要に応じて、表示範囲外の古いデータを削除
        if len(self.x_data_points) > self.max_data:
            self.x_data_points.pop(0)
            self.y_data_points.pop(0)

        self.data_points_item.setData(self.x_data_points, self.y_data_points)

        self.logger.info(f"追加データ: X={x}, Y={y}")

    @Slot()  # 引数なしに変更 (データは共有メモリから読み込むため)
    def updateSmoothedLine(self):
        # --- 共有メモリからの読み込み ---
        # 読み込み中は共有メモリをロックする
        if not self.shm_x.lock():
            self.logger.error(f"Failed to lock shared memory for X (graph): {self.shm_x.errorString()}")
            return
        if not self.shm_y.lock():
            self.logger.error(f"Failed to lock shared memory for Y (graph): {self.shm_y.errorString()}")
            self.shm_x.unlock()  # ロックが一つでも失敗したら、もう片方もアンロック
            return

        try:
            # 共有メモリからデータを読み込み、NumPy配列として再構成
            # shm_data_x/y はすでにビューなので、それ自体が共有メモリを指している
            # 実際に使用する範囲をスライスして取得

            # 現在データがどこまで入っているかを知るために、self.count (メインスレッドのカウンタ) を使う必要がある
            # または、共有メモリに別途「現在の有効なデータ点数」を書き込むフィールドを用意する
            # 今回はExampleクラスのself.countが最新のデータ数を表すので、それを利用する
            current_data_count = QApplication.instance()._main_window_instance.count  # メインウィンドウへの参照を頑張って取得（簡易的な例）
            if current_data_count < 5:  # スムージングの最低点数
                return

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
        # アプリケーション終了時に共有メモリをデタッチ
        # コンシューマーであるグラフ側はdetach()のみ
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

        # データの最大数とカウンタ
        self.max_data = 180  # テスト用、実際の19800点もこの値で制御
        self.count = 0

        self.chart = TrendGraph(self.max_data)
        self.setCentralWidget(self.chart)

        self.data_generator_thread = ThreadDataGenerator(self.max_data)  # 変数名を明確化
        self.data_generator_thread.threadReady.connect(self.on_data_generator_thread_ready)

        # ★重要変更★ workerからのシグナル接続
        self.data_generator_thread.worker.notifyNewData.connect(self.chart.addPoints)
        # スムージングライン更新シグナルは引数なしで接続
        self.data_generator_thread.worker.notifySmoothLineReady.connect(self.chart.updateSmoothedLine)

        self.data_generator_thread.start()

        # メインウィンドウからQApplicationインスタンスを通じて自分自身を保存（TrendGraphからの参照用）
        QApplication.instance()._main_window_instance = self

        # リアルタイム更新のためのQTimer (データ生成をトリガー)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(100)  # テスト用に速く (100ms)

    def closeEvent(self, event: QCloseEvent):
        # タイマー停止
        if self.timer.isActive():
            self.timer.stop()

        # スレッドの安全な終了
        if self.data_generator_thread.isRunning():
            self.logger.info("Stopping data generator thread...")
            self.data_generator_thread.quit()
            self.data_generator_thread.wait()
            self.logger.info("The data generator thread safely terminated.")

        # 共有メモリのデタッチはWorkerとTrendGraphの__del__で行われる
        # __del__が呼ばれるタイミングはPythonのガベージコレクションに依存するので、
        # ここで明示的にdetatch()を呼ぶことも可能だが、QSharedMemoryはプロセス終了時に
        # 自動的にリソースを解放するため、基本的には不要。
        # ただし、create()した側がdetach()を呼んでからquit()/wait()するのがより確実な解放手順。

        super().closeEvent(event)  # QMainWindowのcloseEventを呼び出す
        event.accept()

    @Slot()  # Slotデコレータを追加
    def on_data_generator_thread_ready(self):
        self.logger.info("Data generator thread is ready!")

    @Slot()  # Slotデコレータを追加
    def update_chart(self):
        if self.count >= self.max_data:
            self.timer.stop()
            self.logger.info("リアルタイム更新が終了しました。")
            return

        # ワーカーにデータ生成をリクエスト
        self.data_generator_thread.requestNewData.emit(self.count)
        self.count += 1


if __name__ == "__main__":
    pg.setConfigOption('background', 'k')  # 黒背景 (ダークモード風)
    pg.setConfigOption('foreground', 'w')  # 白前景 (テキストなど)

    app = QApplication(sys.argv)
    window = Example()
    window.show()
    sys.exit(app.exec())
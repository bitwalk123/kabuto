import numpy as np
import time

# データ構造の定義（例: タイムスタンプ、価格、ボリューム）
# あなたの実際のデータに合わせて dtype を調整してください
# f8: float64, i4: int32, S10: 10文字の文字列など
# 例: タイムスタンプ(float64)、価格(float64)、ボリューム(int32)
# デイトレードであれば、通常はより詳細な時間情報や複数の価格（始値、高値、安値、終値）などがあるでしょう
data_dtype = [('timestamp', 'f8'), ('price', 'f8'), ('volume', 'i4')]

# 最大データ点数
max_data_points = 19800

# NumPy 配列を事前に確保
# initial_data = np.zeros(max_data_points, dtype=data_dtype) # 全て0で初期化
# または、より高速な
initial_data = np.empty(max_data_points, dtype=data_dtype) # メモリを確保するだけで初期化しない

current_data_count = 0 # 現在格納されているデータ点数を追跡するカウンター

def add_realtime_data(timestamp, price, volume):
    global current_data_count, initial_data

    if current_data_count < max_data_points:
        # 配列の該当インデックスに直接データを代入
        initial_data[current_data_count] = (timestamp, price, volume)
        current_data_count += 1
    else:
        # 最大点数に達した場合の処理（例: エラーログ、一番古いデータを上書き、あるいは何もせず無視）
        # デイトレードであれば、セッションが終了する、あるいは新しい日/セッションが始まるまで待つ
        # などのロジックが必要になるかもしれません。
        # print("Warning: Maximum list_item points reached. No more list_item added.")
        pass # 今回は単純に無視する例

# シミュレーションの実行例
start_time = time.perf_counter()

for i in range(19800): # 最大点数までデータを追加
    # リアルタイムデータのシミュレーション（実際のデータに合わせて変更）
    current_timestamp = time.time()
    current_price = 100.0 + np.random.randn() * 0.1
    current_volume = np.random.randint(1, 100)
    add_realtime_data(current_timestamp, current_price, current_volume)

end_time = time.perf_counter()
print(f"Adding 19800 data points took: {end_time - start_time:.6f} seconds")

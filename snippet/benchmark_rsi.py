import time
import numpy as np

from modules.technical import RSI


def generate_tick_data(n_ticks: int, initial_price: float = 100.0, volatility: float = 0.01):
    """ティックデータを生成（ランダムウォーク）"""
    np.random.seed(42)
    changes = np.random.normal(0, volatility, n_ticks)
    prices = initial_price + np.cumsum(changes)
    return prices


def benchmark_rsi(window_size: int = 300, n_ticks: int = 100000):
    """RSIクラスのベンチマーク"""
    print(f"{'=' * 60}")
    print(f"RSI Benchmark - Wilder's Method (EMA版)")
    print(f"{'=' * 60}")
    print(f"Window size: {window_size}")
    print(f"Number of ticks: {n_ticks:,}")
    print()

    # ティックデータを生成
    print("Generating tick data...")
    tick_data = generate_tick_data(n_ticks)
    print(f"Generated {len(tick_data):,} ticks")
    print()

    # RSIインスタンスを作成
    rsi = RSI(window_size=window_size)

    # ウォームアップ
    print("Warming up...")
    for i in range(min(1000, n_ticks)):
        rsi.update(tick_data[i])
    rsi.clear()
    print()

    # ベンチマーク実行
    print("Running benchmark...")
    start_time = time.perf_counter()

    for price in tick_data:
        rsi_value = rsi.update(price)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # 結果表示
    print(f"{'=' * 60}")
    print(f"Results:")
    print(f"{'=' * 60}")
    print(f"Total time: {elapsed_time:.4f} seconds")
    print(f"Time per tick: {(elapsed_time / n_ticks) * 1_000_000:.2f} µs")
    print(f"Throughput: {n_ticks / elapsed_time:,.0f} ticks/second")
    print()
    print(f"Final RSI value: {rsi.getValue():.2f}")
    print(f"Final RSI slope: {rsi.getSlope():.2f}")
    print(f"{'=' * 60}")
    print()

    # メモリ使用量の推定
    import sys
    memory_bytes = (
            sys.getsizeof(rsi.avg_gain) +
            sys.getsizeof(rsi.avg_loss) +
            sys.getsizeof(rsi.rsi) +
            sys.getsizeof(rsi.prev_rsi) +
            sys.getsizeof(rsi.prev_value) +
            sys.getsizeof(rsi.init_gain_sum) +
            sys.getsizeof(rsi.init_loss_sum) +
            sys.getsizeof(rsi.init_count)
    )
    print(f"Estimated memory usage: {memory_bytes:,} bytes (~{memory_bytes / 1024:.2f} KB)")
    print()

    return elapsed_time, rsi


def benchmark_comparison(window_size: int = 300, n_ticks: int = 100000):
    """異なるティック数でのベンチマーク比較"""
    print(f"\n{'=' * 60}")
    print(f"Benchmark Comparison (window_size={window_size})")
    print(f"{'=' * 60}\n")

    tick_counts = [10000, 50000, 100000, 500000, 1000000]

    results = []
    for n in tick_counts:
        if n > n_ticks:
            break

        tick_data = generate_tick_data(n)
        rsi = RSI(window_size=window_size)

        start_time = time.perf_counter()
        for price in tick_data:
            rsi.update(price)
        elapsed_time = time.perf_counter() - start_time

        throughput = n / elapsed_time
        results.append((n, elapsed_time, throughput))

        print(f"{n:>10,} ticks: {elapsed_time:>8.4f}s ({throughput:>12,.0f} ticks/sec)")

    print()


if __name__ == "__main__":
    # メインベンチマーク
    benchmark_rsi(window_size=300, n_ticks=100000)

    # 比較ベンチマーク
    benchmark_comparison(window_size=300, n_ticks=1000000)
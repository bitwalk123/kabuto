import datetime
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_path
from modules.agent import PPOAgentSB3
from structs.res import AppRes


def plot_metric(df: pd.DataFrame, path_png: str):
    # x軸：イテレーション（time/iterations）
    x = df["time/iterations"]

    # 可視化したい指標（必要に応じて追加・変更）
    metrics = ["train/approx_kl", "train/clip_fraction", "train/value_loss"]

    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 12

    fig = plt.figure(figsize=(8, 8))
    n = 3
    ax = dict()
    gs = fig.add_gridspec(
        n, 1, wspace=0.0, hspace=0.0, height_ratios=[1 for i in range(n)]
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    # プロット設定
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax[i].plot(x, df[metric], label=metric)
            ax[i].set_xlabel("Iterations")
            ax[i].set_ylabel(metric)
            if i == 2:
                ax[i].set_yscale("log")

    ax[0].set_title("Training Metrics Over Iterations")

    plt.tight_layout()
    plt.savefig(path_png)


if __name__ == "__main__":
    pattern = re.compile(r".+(\d{8})\.xlsx")

    res = AppRes()
    agent = PPOAgentSB3()

    dt = datetime.datetime.now()
    date_str = f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"

    # 学習用データ
    code = "7011"
    # list_file = ["ticks_20250819.xlsx"]
    list_file = sorted(os.listdir(res.dir_collection))
    flag_new_model = True
    for file in list_file:
        print(f"学習するティックデータ : {file}")
        # Excel ファイルのフルパス
        path_excel = get_collection_path(res, file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)
        path_model = get_ppo_model_path(res, code)
        # ログ出力先ディレクトリ（例: logs/ticks_20250819）
        if m := pattern.match(file):
            file_date_str = m.group(1)
        else:
            file_date_str = "unknown"
        log_dir: str = os.path.join(res.dir_log, code, date_str, file_date_str)
        # モデルの学習
        agent.train(df, path_model, log_dir, new_model=flag_new_model)
        if flag_new_model:
            flag_new_model = False

        path_csv = os.path.join(log_dir, "progress.csv")
        path_png = os.path.join(log_dir, "progress.png")
        df = pd.read_csv(path_csv)

        plot_metric(df, path_png)

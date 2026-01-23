import os
import pandas as pd
from funcs.ios import get_excel_sheet
from funcs.logs import setup_logging
from funcs.setting import load_setting
from funcs.tide import get_date_str_from_file, get_intraday_timestamp
from structs.res import AppRes
from widgets.explorer import Explorer


def run_condition(setting, df, code, dict_ts):
    """DOE 条件 1 つ分のシミュレーションを実行して結果を返す"""
    app = Explorer(code, dict_ts)
    app.run(setting, df)
    return app.getTransaction()


if __name__ == "__main__":
    main_logger = setup_logging()
    res = AppRes()

    # list_doe = ["doe-13h"]
    #list_doe = ["doe-14d"]
    list_doe = ["doe-15"]
    for name_doe in list_doe:
        list_code = ["285A", "7011", "7203", "8306"]
        for code in list_code:
            # 出力ディレクトリ
            path_dir = os.path.join(res.dir_output, name_doe, code)
            os.makedirs(path_dir, exist_ok=True)

            # DOE 水準表
            path_doe = os.path.join(res.dir_doe, f"{name_doe}.csv")
            df_matrix = pd.read_csv(path_doe)
            factor_doe = list(df_matrix.columns)

            # 基本設定
            base_setting = load_setting(res, code)

            # 収集ディレクトリ内のファイルを新しい順に処理
            files = sorted(os.listdir(res.dir_collection), reverse=True)
            for excel in files:
                # Excel ファイル名から日付文字列を抽出
                date_str = get_date_str_from_file(excel)
                path_result = os.path.join(path_dir, f"result_{date_str}.csv")

                # すでに結果があるならスキップ
                if os.path.exists(path_result):
                    continue

                # タイムスタンプ取得
                dict_ts = get_intraday_timestamp(excel)

                # ティックデータ読み込み
                path_excel = os.path.join(res.dir_collection, excel)
                df = get_excel_sheet(path_excel, code)
                # データがなければスキップ
                if len(df) == 0:
                    continue

                # 結果格納用辞書を初期化
                dict_doe = {key: [] for key in ["file", "code", "trade", "total"] + factor_doe}

                # DOE 条件ループ
                for idx, row in df_matrix.iterrows():
                    print(f"\n実験条件# {idx}")

                    # 設定をコピーして DOE 条件を適用
                    setting = base_setting.copy()
                    for key in factor_doe:
                        setting[key] = row[key]

                    # シミュレーション実行
                    _, n_trade, total = run_condition(setting, df, code, dict_ts)
                    print(f"取引回数: {n_trade} 回 / 総収益: {total} 円/100株")

                    # 結果を保存
                    dict_doe["file"].append(excel)
                    dict_doe["code"].append(code)
                    dict_doe["trade"].append(n_trade)
                    dict_doe["total"].append(total)

                    for key in factor_doe:
                        dict_doe[key].append(row[key])

                # 1 ファイル分の DOE 結果を保存
                df_doe = pd.DataFrame(dict_doe)
                df_doe.to_csv(path_result, index=False)
                print(f"\n{date_str} の結果を {path_result} に保存しました。")

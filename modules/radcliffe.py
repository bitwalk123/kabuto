import os

from funcs.ios import get_excel_sheet
from funcs.setting import load_setting
from funcs.tide import get_date_str_from_file
from modules.agent import CronAgent
from structs.res import AppRes


class Radcliffe:
    """
    同一条件で通しでシミュレーションを実施
    """

    def __init__(self):
        self.res = res = AppRes()

        # プレフックス
        self.prefix = "por"

        # 対象銘柄
        self.code = code = "7011"
        # self.code = code = "7203"
        # self.code = code = "8306"

        # 銘柄別の設定を読み込む
        self.dict_setting = load_setting(self.res, code)

        self.dir_transaction = dir_transaction = os.path.join(
            res.dir_transaction, code
        )
        os.makedirs(dir_transaction, exist_ok=True)

        # GUI 無しエージェントのインスタンス
        self.agent = CronAgent(code)

    def run(self):
        files = sorted(os.listdir(self.res.dir_collection))

        for excel in files:
            path_excel = os.path.join(self.res.dir_collection, excel)
            # 指定したシート (= code) を読み込む
            df = get_excel_sheet(path_excel, self.code)
            self.agent.run(self.dict_setting, df)
            df_transaction = self.agent.getTransaction()
            print(df_transaction)
            date_str = get_date_str_from_file(excel)
            file_transaction = f"{self.prefix}_{date_str}.csv"
            out_transaction = os.path.join(
                self.dir_transaction,
                file_transaction
            )
            df_transaction.to_csv(out_transaction, index=False)
            print(f"{out_transaction} に保存しました。")


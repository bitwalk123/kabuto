# Apostle が算出したシミュレーション結果を集計するクラス
import glob
import os

from structs.res import AppRes


class Lagrange:
    def __init__(self):
        self.res = AppRes()

    def run(self):
        list_csv = glob.glob(os.path.join(self.res.dir_report, "*", "report_*.csv"))
        for csv in list_csv[:1]:
            print(csv)
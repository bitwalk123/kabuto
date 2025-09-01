from funcs.ios import get_excel_sheet
from modules.rl_ppo_lite_20250901_1 import Trainer

if __name__ == "__main__":
    code = "7011"
    list_excel = [
        "excel/tick_20250819.xlsx",
        "excel/tick_20250820.xlsx",
        "excel/tick_20250821.xlsx",
        "excel/tick_20250822.xlsx",
        "excel/tick_20250825.xlsx",
        "excel/tick_20250826.xlsx",
        "excel/tick_20250827.xlsx",
        "excel/tick_20250828.xlsx",
        "excel/tick_20250829.xlsx",
    ]
    for epoch in range(10):
        for file_excel in list_excel:
            df = get_excel_sheet(file_excel, code)

            trainer = Trainer()
            earnings = trainer.train(df)
            print(f"Epoch: {epoch}, {file_excel}, 収益：{earnings}")

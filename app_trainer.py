from funcs.ios import get_excel_sheet
from modules.rl_ppo_lite_20250829 import Trainer

if __name__ == "__main__":
    code = "7011"
    list_excel = ["excel/tick_20250826.xlsx", "excel/tick_20250828.xlsx"]
    for epoch in range(10):
        for file_excel in list_excel:
            df = get_excel_sheet(file_excel, code)

            trainer = Trainer(model_path="models/ppo_7011_20250829.pch", feature_n=60, device="cpu")
            trainer.train(df)

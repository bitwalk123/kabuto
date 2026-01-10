from funcs.logs import setup_logging
from funcs.setting import load_setting
from modules.disciple import Disciple
from structs.res import AppRes

if __name__ == "__main__":
    main_logger = setup_logging()
    res = AppRes()

    excel = "ticks_20260109.xlsx"
    code = "7011"
    dict_setting = load_setting(res, code)

    app = Disciple(excel, code, dict_setting)
    app.run()

    # 取引明細
    df_transaction = app.getTransaction()
    print("\n取引明細")
    print(df_transaction)
    n_trade = len(df_transaction)
    total = df_transaction["損益"].sum()
    print(f"取引回数: {n_trade} 回 / 総収益: {total} 円/株")

    # テクニカル指標
    print("\nテクニカル指標")
    df_technical = app.getTechnicals()
    print(df_technical)

    # 観測値
    print("\n観測値")
    df_obs = app.getObservations()
    print(df_obs)

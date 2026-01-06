from funcs.logs import setup_logging
from modules.disciple import Disciple

if __name__ == "__main__":
    main_logger = setup_logging()
    excel = "ticks_20260105.xlsx"
    code = "7011"

    app = Disciple(excel, code)
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

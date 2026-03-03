import numpy as np
import pandas as pd


def calc_robust_bollinger(df: pd.DataFrame, period: int):
    """
    メジアン統計によるボリンジャーバンドの算出
    :param df:
    :param period:
    :return:
    """
    r_last = len(df)
    r1 = 0
    df['Median'] = np.nan
    df['Q1'] = np.nan
    df['Q3'] = np.nan
    df['Lower'] = np.nan
    df['Upper'] = np.nan
    while r1 < r_last - period:
        r2 = r1 + period
        df1 = df.iloc[r1:r2].copy()
        med = np.median(df1['Close'])
        q3, q1 = np.percentile(df1['Close'], [75, 25])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)  # 下限を設定
        upper_bound = q3 + (1.5 * iqr)  # 上限を設定
        name_index = df.index[r2]
        df.at[name_index, 'Median'] = med
        df.at[name_index, 'Q1'] = q1
        df.at[name_index, 'Q3'] = q3
        df.at[name_index, 'Lower'] = lower_bound
        df.at[name_index, 'Upper'] = upper_bound
        r1 += 1


def psar(ohlc: pd.DataFrame, iaf: float = 0.02, maxaf: float = 0.2) -> dict:
    length = len(ohlc)
    high = ohlc['High'].tolist()
    low = ohlc['Low'].tolist()
    close = ohlc['Close'].tolist()

    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length

    bull = True
    af = iaf
    ep = low[0]
    price_high = high[0]
    price_low = low[0]

    for i in range(2, length):
        if bull:
            psar[i] = psar[i - 1] + af * (price_high - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (price_low - psar[i - 1])
        reverse = False

        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = price_high
                price_low = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = price_low
                price_high = high[i]
                af = iaf

        if not reverse:
            if bull:
                if high[i] > price_high:
                    price_high = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < price_low:
                    price_low = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]

        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]

    return {
        'bear': np.array(psarbear, dtype='float64'),
        'bull': np.array(psarbull, dtype='float64'),
    }


def calculate_psar(ohlc: pd.DataFrame, iaf: float = 0.02, maxaf: float = 0.2, af_step: float = 0.02) -> dict:
    length = len(ohlc)
    if length < 2:
        nan_array = np.full(length, np.nan, dtype=np.float64)
        return {'bear': nan_array, 'bull': nan_array}

    high = ohlc['High'].to_numpy()
    low = ohlc['Low'].to_numpy()
    close = ohlc['Close'].to_numpy()

    sar_values = np.full(length, np.nan, dtype=np.float64)
    ep_values = np.full(length, np.nan, dtype=np.float64)  # デバッグ用
    af_values = np.full(length, np.nan, dtype=np.float64)  # デバッグ用

    psarbull = np.full(length, np.nan, dtype=np.float64)
    psarbear = np.full(length, np.nan, dtype=np.float64)

    # --- 初期設定 (最初の有効なSARはインデックス1から) ---
    # トレンドの初期決定: 最初の2日間の終値で判断
    if close[1] > close[0]:
        bull = True  # 上昇トレンド
        current_ep = high[1]  # EPは最高値
        current_sar = low[0]  # SARの初期値は前日の安値
    else:
        bull = False  # 下降トレンド
        current_ep = low[1]  # EPは最安値
        current_sar = high[0]  # SARの初期値は前日の高値

    current_af = iaf  # 加速因子

    # インデックス1のSAR値を格納
    sar_values[1] = current_sar
    ep_values[1] = current_ep
    af_values[1] = current_af

    if bull:
        psarbull[1] = current_sar
    else:
        psarbear[1] = current_sar

    print(f"--- PSAR初期設定 (idx 1) ---")
    print(f"  bull: {bull}, current_sar: {current_sar:.2f}, current_ep: {current_ep:.2f}, current_af: {current_af:.3f}")

    # --- ループ開始 (インデックス 2 から) ---
    for i in range(2, length):
        prev_sar = sar_values[i - 1]
        prev_ep = ep_values[i - 1]
        prev_af = af_values[i - 1]

        # 暫定的な今日のSARを計算 (トレンド継続と仮定)
        if bull:  # 上昇トレンドの場合
            calculated_sar_today = prev_sar + prev_af * (prev_ep - prev_sar)
            # キャッピングルール: SARは現在の安値または前日の安値より高くなってはならない
            # PSARが価格を上抜けた場合は、今日の安値か前日の安値にSARをセット
            if calculated_sar_today > low[i]:
                calculated_sar_today = min(calculated_sar_today, low[i])  # ワイルダーのルールは現在の安値と前日の安値の低い方
            if calculated_sar_today > low[i - 1]:
                calculated_sar_today = min(calculated_sar_today, low[i - 1])


        else:  # 下降トレンドの場合
            calculated_sar_today = prev_sar - prev_af * (prev_sar - prev_ep)
            # キャッピングルール: SARは現在の高値または前日の高値より低くなってはならない
            # PSARが価格を下抜けた場合は、今日の高値か前日の高値にSARをセット
            if calculated_sar_today < high[i]:
                calculated_sar_today = max(calculated_sar_today, high[i])  # ワイルダーのルールは現在の高値と前日の高値の高い方
            if calculated_sar_today < high[i - 1]:
                calculated_sar_today = max(calculated_sar_today, high[i - 1])

        # トレンド反転のチェック (終値がSARをクロスした場合)
        reversed_this_period = False
        next_bull = bull  # 次の日のトレンド方向を一時的に保持
        next_sar = current_sar  # 次の日のSARを一時的に保持
        next_ep = current_ep  # 次の日のEPを一時的に保持
        next_af = current_af  # 次の日のAFを一時的に保持

        if bull:  # 現在上昇トレンド
            if close[i] < calculated_sar_today:  # 終値がSARを下抜けた場合、トレンド反転
                reversed_this_period = True
                next_bull = False  # 新しいトレンドは下降トレンド
                next_sar = prev_ep  # 新しいSARは、前のトレンドのEP (最高値)
                next_ep = low[i]  # 新しいEPは、現在の安値
                next_af = iaf  # 加速因子をリセット
        else:  # 現在下降トレンド
            if close[i] > calculated_sar_today:  # 終値がSARを上抜けた場合、トレンド反転
                reversed_this_period = True
                next_bull = True  # 新しいトレンドは上昇トレンド
                next_sar = prev_ep  # 新しいSARは、前のトレンドのEP (最安値)
                next_ep = high[i]  # 新しいEPは、現在の高値
                next_af = iaf  # 加速因子をリセット

        # トレンドが継続する場合のSAR、EP、AFの更新
        if not reversed_this_period:
            next_sar = calculated_sar_today  # 反転しなかった場合は、計算したSARを使用
            if bull:  # 上昇トレンド継続
                # EPはトレンド中の最高値
                if high[i] > prev_ep:
                    next_ep = high[i]
                    next_af = min(prev_af + af_step, maxaf)  # AFを加速
                else:
                    next_ep = prev_ep
                    next_af = prev_af
            else:  # 下降トレンド継続
                # EPはトレンド中の最安値
                if low[i] < prev_ep:
                    next_ep = low[i]
                    next_af = min(prev_af + af_step, maxaf)  # AFを加速
                else:
                    next_ep = prev_ep
                    next_af = prev_af

        # 最終的な今日のSAR、EP、AFを配列に格納
        sar_values[i] = next_sar
        ep_values[i] = next_ep
        af_values[i] = next_af
        bull = next_bull  # bullフラグを更新

        # SARを対応するbull/bear配列に格納
        if bull:
            psarbull[i] = sar_values[i]
        else:
            psarbear[i] = sar_values[i]

        # デバッグ出力
        """
        print(f"Date: {ohlc.index[i].strftime('%Y-%m-%d')}, Close: {close[i]:.2f}")
        print(f"  bull: {bull}, reversed: {reversed_this_period}")
        print(f"  SAR: {sar_values[i]:.2f}, EP: {ep_values[i]:.2f}, AF: {af_values[i]:.3f}")
        print(f"  psarbull[i]: {psarbull[i]:.2f} (if not nan), psarbear[i]: {psarbear[i]:.2f} (if not nan)\n")
        """

    return {
        'bear': psarbear,
        'bull': psarbull,
    }

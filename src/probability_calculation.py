import numpy as np
import pandas as pd


def calculate_rank_probability(df):
    """
    1着確率を元に、2着と3着の確率を計算する
    計算方法にはベンターが提唱する計算式を使用する

    Parameters
    ----------
    df : pandas.DataFrame
        レースごとの1着確率を保持しているデータフレーム。

    Returns
    -------
    df : pandas.DataFrame
        レースごとに1着 / 2着 / 3着確率を計算したデータフレーム
    """

    # ここの値はベンターのものなので再計算するのが良さそう
    SECOND_CORRECT_VAL = 0.81
    THIRD_CORRECT_VAL = 0.65

    total = (
        df.groupby("race_id")
        .agg({"predict": np.sum})
        .rename(columns={"predict": "total_proba"})
    )
    base = pd.merge(df, total, on=["race_id"], how="left")
    base["first_proba"] = base["predict"] / base["total_proba"]

    base["second_proba_source"] = base["first_proba"] ** SECOND_CORRECT_VAL
    base["third_proba_source"] = base["first_proba"] ** THIRD_CORRECT_VAL

    source = (
        base.groupby("race_id")
        .agg({"second_proba_source": np.sum, "third_proba_source": np.sum})
        .rename(
            columns={
                "second_proba_source": "total_second_proba_source",
                "third_proba_source": "total_third_proba_source",
            }
        )
    )

    base = pd.merge(base, source, on=["race_id"], how="left")
    base["second_proba"] = (
        base["second_proba_source"] / base["total_second_proba_source"]
    )
    base["third_proba"] = base["third_proba_source"] / base["total_third_proba_source"]

    return base[["race_id", "horse_no", "first_proba", "second_proba", "third_proba"]]


def correct_proba_in_race(df):
    """
    レース毎に勝率を補正する

    Parameters
    ----------
    df : pandas.DataFrame
        データフレーム

    Returns
    -------
    df : pandas.DataFrame
        補正後のデータフレーム
    """
    total = (
        df.groupby("race_id")
        .agg({"proba": np.sum})
        .rename(columns={"proba": "total_proba"})
    )
    base = pd.merge(df, total, on=["race_id"], how="left")
    base["correct_proba"] = base["proba"] / base["total_proba"]

    return base.drop(["total_proba"], axis=1)


def create_horse_proba_array(df, proba_key):
    """
    馬の予測値の配列を作る。馬番が欠番していた場合はnanで埋める。

    Parameters
    ----------
    df : pandas.DataFrame
        データフレーム
    proba_key : string
        予測カラムの名前

    Returns
    -------
    array : array
        予測値の配列
    """

    array = []
    horse_numbers = df.horse_no.values
    values = df[proba_key].values
    current_horse_no = 1

    for i in range(len(horse_numbers)):
        no = horse_numbers[i]
        if no == current_horse_no:
            # 抜け番なし
            array.append(values[i])
        else:
            # 抜け番あり
            for j in range(no - current_horse_no):
                # 抜けてる数だけ埋める
                array.append(np.nan)
                current_horse_no += 1

            array.append(values[i])

        current_horse_no += 1

    return array

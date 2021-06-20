from sklearn.preprocessing import StandardScaler


def standard_scaler_by_race(df, standard_cols):
    """
    レース毎にカラムを標準化する
    Parameters
    ----------
    df : pandas.DataFrame
        データフレーム
        race_id必須
    standard_cols : array
        標準化対象のカラム
    """
    race_ids = df.race_id.unique()
    transformed = []

    for race_id in race_ids:
        target_race = df[df["race_id"] == race_id]
        num_datas = target_race[standard_cols]

        sc = StandardScaler()
        sc.fit(num_datas)
        transformed.extend(sc.transform(num_datas))

    df[standard_cols] = transformed

    return df

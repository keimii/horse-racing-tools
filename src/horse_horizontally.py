import pandas as pd


class HorseHorizontally(object):
    """
    レース単位で馬の特徴量を横並べにする
    """

    def __init__(self, df, keys, columns, max_horizontal_no=18):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            データフレーム
            race_id, horse_no必須
        keys : array
            レコード毎に保持するキー
        columns : array
            横並べにするカラム
        max_horizontal_no : int
            最大番号
        """
        self.df = df
        self.keys = keys
        self.columns = columns
        self.max_horizontal_no = max_horizontal_no

    def line_up_by_horse_no_order(self):
        """
        レース毎に馬の特徴量を18頭立てで横並べにする
        18に満たない場合はNoneで埋める
        """
        key_df = self.df[self.keys]

        for i in range(self.max_horizontal_no):
            horse_no = i + 1
            horse_no_df = self.df[self.df.horse_no == horse_no][
                ["race_id"] + self.columns
            ]
            rename_columns = self._column_name(horse_no)
            key_df = pd.merge(
                key_df,
                horse_no_df.rename(columns=rename_columns),
                on="race_id",
                how="left",
            )

        return key_df

    def _column_name(self, no):
        column_dict = dict()
        for column in self.columns:
            column_dict[column] = "no{}_{}".format(no, column)

        return column_dict

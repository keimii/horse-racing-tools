import numpy as np


class Raffine(object):
    """
    ラフィーネ法に基づいて払戻均等法の賭け率を計算する
    """

    def __init__(self, df, threshold=1.5, filter_key="expectation"):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            データフレーム
        threshold : float
            合成オッズの閾値
        filter_key : string
            合成オッズが閾値以下だった場合に足切りして再計算するためのキー
        """

        self.df = df
        self.threshold = threshold
        # 合成オッズ
        self.synthetic_odds = None
        # ラフィーネ法で期待値割れした時に足切りに使うカラム名
        self.filter_key = filter_key

    def calculate(self):
        """
        ラフィーネ法で計算し、対象となる馬券を賭け率ともに返却する

        Returns
        -------
        df : pandas.DataFrame
            賭け対象のデータフレーム
        """

        filtered = self._filter_raffine(self.df.copy())

        if filtered.shape[0] == 0:
            return filtered

        min_odds = min(filtered.odds.values)
        coef = min_odds * 100

        values = (coef / filtered.odds).apply(np.floor)
        sum_values = sum(values)

        self.synthetic_odds = coef / sum_values

        filtered["raffine_coef"] = values / sum_values

        return filtered

    def _filter_raffine(self, target_df):
        if target_df.shape[0] == 0:
            return target_df

        min_odds = min(target_df.odds.values)
        coef = min_odds * 100

        values = (coef / target_df.odds).apply(np.floor)
        sum_values = sum(values)

        synthetic_odds = coef / sum_values

        if synthetic_odds >= self.threshold:
            return target_df

        next_num = target_df.shape[0] - 1

        return self._filter_raffine(
            target_df.sort_values(self.filter_key, ascending=False).head(next_num)
        )

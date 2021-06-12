import math

import numpy as np
import pandas as pd

from src.raffine import Raffine


class KellySimpleFormula(object):
    """
    ケリーの公式に基づいて適切な掛け率を計算する
    """

    @classmethod
    def calculate(cls, win_proba, odds, coef):
        """
        Parameters
        ----------
        win_proba : float
            予想勝利確率
        odds : float
            オッズ
        coef : int
            ハーフケリーにしたい場合、2を指定する

        Returns
        -------
        rate : float
            賭け率
        """

        bet_rate = win_proba / odds

        if bet_rate < 0:
            return 0

        return math.ceil((bet_rate / coef) * 1000) / 1000


class KellyFormula(object):
    """
    ケリーの公式に基づいて適切な掛け率を計算する
    """

    @classmethod
    def calculate(cls, win_proba, odds, coef):
        """
        Parameters
        ----------
        win_proba : float
            予想勝利確率
        odds : float
            オッズ
        coef : int
            ハーフケリーにしたい場合、2を指定する

        Returns
        -------
        rate : float
            賭け率
        """

        profit = odds - 1
        loss = -1
        lose_proba = 1 - win_proba

        edge = (profit * win_proba) + (loss * lose_proba)
        bet_rate = edge / profit

        if bet_rate < 0:
            return 0

        return math.ceil((bet_rate / coef) * 1000) / 1000


class Simulator(object):
    """
    ケリーの公式を使ってシミュレーションする
    """

    def __init__(
        self,
        df,
        payoff,
        initial_balance,
        coef,
        kelly_type="kelly_simple_formula",
        show_debug=False,
    ):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            データフレーム
        payoff : pandas.DataFrame
            払い戻し
        initial_balance : float
            初期残高
        coef : int
            ハーフケリーにしたい場合、2を指定する
        kelly_type : string
            かけ率計算に使用する計算種類
        show_debug : Bool
            デバッグ用のログを表示したい場合、True
        """
        self.df = df
        self.payoff = payoff
        self.initial_balance = initial_balance
        self.coef = coef
        self.show_debug = show_debug
        self.kelly_type = kelly_type

    def run(self):
        current_amount = self.initial_balance
        result = []
        p_idx = 1
        size = len(self.df.race_id.unique())

        for race_id in self.df.race_id.unique():
            if self.show_debug:
                print(
                    "\r race_id: {0}, status: {1}/{2}".format(race_id, p_idx, size),
                    end="",
                )

            race_df = self.df[self.df.race_id == race_id]
            held_in = race_df.held_in.values[0]

            rf_ret = self._calculate_purchase_amount(race_df, current_amount)

            if (rf_ret is None) | (current_amount <= 0):
                result.append([race_id, held_in, 0, 0, 0, 0, 0, current_amount])
                p_idx += 1
                continue

            merged = pd.merge(
                rf_ret,
                self.payoff,
                on=["race_id", "horse_no1", "horse_no2"],
                how="left",
            )
            merged["total_payoff"] = merged["payoff"].fillna(0) * merged["bet_rate"]

            ticket_num = merged.shape[0]
            total_purchase_amount = merged.bet_rate.sum() * 100
            total_payoff_amount = merged.total_payoff.sum()
            hit_num = merged[merged.payoff > 0].shape[0]
            income = total_payoff_amount - total_purchase_amount

            current_amount = current_amount + income

            result.append(
                [
                    race_id,
                    held_in,
                    ticket_num,
                    hit_num,
                    total_purchase_amount,
                    total_payoff_amount,
                    income,
                    current_amount,
                ]
            )

            p_idx += 1

        return result

    def _calculate_purchase_amount(self, df, current_amount):
        rf = Raffine(df)
        rf_ret = rf.calculate()

        # 合成オッズが期待値を上回る組み合わせが見つからないケース
        if (rf.synthetic_odds is None) | (rf_ret.shape[0] == 0):
            return None

        win_proba = df.correct_proba.sum()
        bet_rate = self._kelly_formula_class().calculate(
            win_proba=win_proba, odds=rf.synthetic_odds, coef=self.coef
        )

        bet_amount = math.floor(current_amount * bet_rate)

        rf_ret["bet_rate"] = (rf_ret["raffine_coef"] * bet_amount / 100).apply(np.floor)

        zero_ticket = rf_ret[rf_ret.bet_rate < 1]

        if zero_ticket.shape[0] > 0:
            # 0円が含まれると的中率と払戻額の関係が壊れるので、0以外のもので計算し直す
            drop_df = df.drop(zero_ticket.index, errors="ignore")
            return self._calculate_purchase_amount(drop_df, current_amount)

        return rf_ret

    def _kelly_formula_class(self):
        if self.kelly_type == "kelly_formula":
            return KellyFormula

        if self.kelly_type == "kelly_simple_formula":
            return KellySimpleFormula

        if self.kelly_type == "kelly_criterion":
            return None

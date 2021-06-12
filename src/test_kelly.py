import unittest

import numpy as np
import pandas as pd

import kelly as kl


class KellyFormulaTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_kelly_formula(self):
        result = kl.KellyFormula.calculate(win_proba=0.5, odds=3, coef=1)

        self.assertEqual(0.25, result)

    def test_half_kelly_formula(self):
        result = kl.KellyFormula.calculate(win_proba=0.5, odds=3, coef=2)

        self.assertEqual(0.125, result)

    def test_minus_expectation(self):
        result = kl.KellyFormula.calculate(win_proba=0.49, odds=3, coef=2)

        self.assertEqual(0.118, result)


class KellySimpleFormulaTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_kelly_formula(self):
        result = kl.KellySimpleFormula.calculate(win_proba=0.5, odds=3, coef=1)

        self.assertEqual(0.167, result)

    def test_half_kelly_formula(self):
        result = kl.KellySimpleFormula.calculate(win_proba=0.5, odds=3, coef=2)

        self.assertEqual(0.084, result)

    def test_minus_expectation(self):
        result = kl.KellySimpleFormula.calculate(win_proba=0.49, odds=2, coef=2)

        self.assertEqual(0.123, result)


class KellyCriterionTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_kelly_formula(self):
        result = kl.KellyCriterion.calculate(win_proba=0.5, odds=3, coef=1)

        self.assertEqual(0.25, result)

    def test_half_kelly_formula(self):
        result = kl.KellyCriterion.calculate(win_proba=0.5, odds=3, coef=2)

        self.assertEqual(0.125, result)

    def test_minus_expectation(self):
        result = kl.KellyCriterion.calculate(win_proba=0.49, odds=2, coef=2)

        self.assertEqual(0, result)


class KellySumulatorTest(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(
            [
                [1, "2021-03-01", 1, 2, 120.5, 0.09, 10.845],
                [1, "2021-03-01", 1, 3, 90.5, 0.05, 4.525],
                [1, "2021-03-01", 1, 4, 60.5, 0.05, 30.25],
                [1, "2021-03-01", 1, 5, 20.5, 0.07, 1.435],
                [2, "2021-03-02", 1, 2, 200.5, 0.12, 24.06],
                [2, "2021-03-02", 1, 3, 85.5, 0.04, 3.4],
                [2, "2021-03-02", 1, 4, 51.5, 0.06, 3.09],
                [2, "2021-03-02", 1, 5, 15.5, 0.2, 9.61],
                [3, "2021-03-02", 1, 2, 1.5, 0.12, 0.18],
            ],
            columns=[
                "race_id",
                "held_in",
                "horse_no1",
                "horse_no2",
                "odds",
                "correct_proba",
                "expectation",
            ],
        )
        payoff_df = pd.DataFrame(
            [
                [1, 1, 5, 2050],
            ],
            columns=["race_id", "horse_no1", "horse_no2", "payoff"],
        )

        self.simullator = kl.Simulator(
            df=df, payoff=payoff_df, initial_balance=1000000, coef=2
        )
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_kelly_formula(self):
        result = self.simullator.run()

        np.testing.assert_array_equal(
            [
                [1, "2021-03-01", 4, 1, 11000.0, 131200.0, 120200.0, 1120200.0],
                [2, "2021-03-02", 4, 0, 23600.0, 0.0, -23600.0, 1096600.0],
                [3, "2021-03-02", 1, 0, 43900.0, 0.0, -43900.0, 1052700.0],
            ],
            result,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd

import raffine as rf


class RaffineTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_normal(self):
        df = pd.DataFrame(
            [
                [22.0, 0.0],
                [24.0, 0.1],
                [28.0, 0.2],
                [28.4, 0.3],
                [33.9, 0.4],
                [34.7, 0.5],
                [34.8, 0.6],
                [38.9, 0.7],
                [41.3, 0.8],
                [42.3, 0.9],
            ],
            columns=["odds", "expectation"],
        )

        raffine = rf.Raffine(df)
        result = raffine.calculate()

        self.assertEqual(3.156, np.round(raffine.synthetic_odds, 3))
        np.testing.assert_array_equal(
            [0.143, 0.131, 0.112, 0.110, 0.092, 0.090, 0.090, 0.080, 0.076, 0.075],
            np.round(result.raffine_coef.values, 3),
        )

    def test_minus_synthetic_odds(self):
        df = pd.DataFrame(
            [
                [1.1, 0.0],
                [1.2, 0.1],
                [4.2, 0.2],
            ],
            columns=["odds", "expectation"],
        )

        raffine = rf.Raffine(df)
        result = raffine.calculate()

        self.assertEqual(4.2, np.round(raffine.synthetic_odds, 3))
        np.testing.assert_array_equal(
            [1.0],
            np.round(result.raffine_coef.values, 3),
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import pandas as pd

import src.probability_calculation as pc


class ProbabilityCalculationTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_calculate_rank_probability(self):
        df = pd.DataFrame(
            [
                [1, 1, 0.1],
                [1, 2, 0.3],
                [1, 3, 0.5],
                [1, 4, 0.7],
                [1, 5, 0.9],
                [2, 1, 0.2],
                [2, 2, 0.4],
                [2, 3, 0.6],
                [2, 4, 0.8],
                [2, 5, 0.10],
            ],
            columns=["race_id", "horse_no", "predict"],
        )

        calc_df = pc.calculate_rank_probability(df)

        np.testing.assert_array_equal(
            [0.040, 0.120, 0.200, 0.280, 0.360, 0.095, 0.190, 0.286, 0.381, 0.048],
            np.round(calc_df.first_proba.values, 3),
        )
        np.testing.assert_array_equal(
            [0.056, 0.136, 0.206, 0.270, 0.332, 0.113, 0.199, 0.276, 0.348, 0.065],
            np.round(calc_df.second_proba.values, 3),
        )
        np.testing.assert_array_equal(
            [0.074, 0.150, 0.209, 0.260, 0.307, 0.130, 0.203, 0.265, 0.319, 0.083],
            np.round(calc_df.third_proba.values, 3),
        )

    def test_correct_proba_in_race(self):
        df = pd.DataFrame(
            [
                [1, 1, 0.1],
                [1, 2, 0.3],
                [1, 3, 0.5],
                [1, 4, 0.7],
                [1, 5, 0.9],
                [2, 1, 0.2],
                [2, 2, 0.4],
                [2, 3, 0.6],
                [2, 4, 0.8],
                [2, 5, 0.10],
            ],
            columns=["race_id", "horse_no", "proba"],
        )

        calc_df = pc.correct_proba_in_race(df)

        np.testing.assert_array_equal(
            [0.040, 0.120, 0.200, 0.280, 0.360, 0.095, 0.190, 0.286, 0.381, 0.048],
            np.round(calc_df.correct_proba.values, 3),
        )

    def test_create_horse_proba_array(self):
        df1 = pd.DataFrame(
            [
                [1, 1, 0.1],
                [1, 2, 0.3],
                [1, 3, 0.5],
                [1, 4, 0.7],
            ],
            columns=["race_id", "horse_no", "proba"],
        )
        df2 = pd.DataFrame(
            [
                [1, 2, 0.1],
                [1, 4, 0.3],
                [1, 7, 0.5],
                [1, 11, 0.7],
            ],
            columns=["race_id", "horse_no", "proba"],
        )

        np.testing.assert_array_equal(
            [0.1, 0.3, 0.5, 0.7], pc.create_horse_proba_array(df1, "proba")
        )

        np.testing.assert_array_equal(
            [
                np.nan,
                0.1,
                np.nan,
                0.3,
                np.nan,
                np.nan,
                0.5,
                np.nan,
                np.nan,
                np.nan,
                0.7,
            ],
            pc.create_horse_proba_array(df2, "proba"),
        )


if __name__ == "__main__":
    unittest.main()

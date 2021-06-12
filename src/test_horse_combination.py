import unittest

import numpy as np

import horse_combination as hc


class HorseCombinationTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_normal(self):
        favorite_horse_numbers = [1, 2]
        rival_horse_numbers = [1, 2, 3]
        first_probas = [0.040, 0.120, 0.200]
        second_probas = [0.056, 0.136, 0.206]
        with_proba = True

        comb_array = hc.quinella_combination(
            race_id=1,
            favorite_horse_numbers=favorite_horse_numbers,
            rival_horse_numbers=rival_horse_numbers,
            first_probas=first_probas,
            second_probas=second_probas,
            with_proba=with_proba,
        )

        np.testing.assert_array_equal(
            [[1, 1, 2, 0.0135], [1, 1, 3, 0.0228], [1, 2, 3, 0.0629]],
            np.round(comb_array, 4),
        )

    def test_without_proba(self):
        favorite_horse_numbers = [1, 2]
        rival_horse_numbers = [1, 2, 3]
        first_probas = [0.040, 0.120, 0.200]
        second_probas = [0.056, 0.136, 0.206]
        with_proba = False

        comb_array = hc.quinella_combination(
            race_id=1,
            favorite_horse_numbers=favorite_horse_numbers,
            rival_horse_numbers=rival_horse_numbers,
            first_probas=first_probas,
            second_probas=second_probas,
            with_proba=with_proba,
        )

        np.testing.assert_array_equal(
            [[1, 1, 2], [1, 1, 3], [1, 2, 3]], np.round(comb_array, 4)
        )


if __name__ == "__main__":
    unittest.main()

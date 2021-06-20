import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from horse_horizontally import HorseHorizontally


class HorseHorizontallyTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_column_name(self):
        hor = HorseHorizontally(None, [], ["proba1", "proba2"])

        self.assertEqual(
            {"proba1": "no1_proba1", "proba2": "no1_proba2"}, hor._column_name(1)
        )

    def test_line_up_by_horse_no_order(self):
        df = pd.DataFrame(
            [
                [1, 1, 0.1, 0.02],
                [1, 2, 0.3, 0.04],
                [1, 3, 0.5, 0.06],
                [2, 1, 0.2, 0.01],
                [2, 2, 0.4, 0.03],
            ],
            columns=["race_id", "horse_no", "proba1", "proba2"],
        )

        hor = HorseHorizontally(df, ["race_id", "horse_no"], ["proba1", "proba2"], 3)

        ret_df = hor.line_up_by_horse_no_order()

        expected_df = pd.DataFrame(
            [
                [1, 1, 0.1, 0.02, 0.3, 0.04, 0.5, 0.06],
                [1, 2, 0.1, 0.02, 0.3, 0.04, 0.5, 0.06],
                [1, 3, 0.1, 0.02, 0.3, 0.04, 0.5, 0.06],
                [2, 1, 0.2, 0.01, 0.4, 0.03, None, None],
                [2, 2, 0.2, 0.01, 0.4, 0.03, None, None],
            ],
            columns=[
                "race_id",
                "horse_no",
                "no1_proba1",
                "no1_proba2",
                "no2_proba1",
                "no2_proba2",
                "no3_proba1",
                "no3_proba2",
            ],
        )
        assert_frame_equal(ret_df, expected_df)


if __name__ == "__main__":
    unittest.main()

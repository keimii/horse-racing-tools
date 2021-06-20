import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from standard_scaler import standard_scaler_by_race


class StandardScalerTest(unittest.TestCase):
    def setUp(self):
        # 初期化処理
        pass

    def tearDown(self):
        # 終了処理
        pass

    def test_standard_scaler_by_race(self):
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

        ret_df = standard_scaler_by_race(df, ["proba1", "proba2"])

        expected_df = pd.DataFrame(
            [
                [1, 1, -1.224744871391589, -1.2247448713915892],
                [1, 2, 0, 0],
                [1, 3, 1.224744871391589, 1.224744871391589],
                [2, 1, -1.0000000000000002, -1.0],
                [2, 2, 0.9999999999999998, 0.9999999999999998],
            ],
            columns=["race_id", "horse_no", "proba1", "proba2"],
        )
        assert_frame_equal(ret_df, expected_df)


if __name__ == "__main__":
    unittest.main()

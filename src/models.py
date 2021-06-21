import lightgbm as lgb
import numpy as np
import optuna.integration.lightgbm as tuner
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold


def split_train_and_validation(dataset, fold):
    race_id = dataset.race_id
    unique_race_ids = race_id.unique()
    kf = KFold(n_splits=fold, shuffle=True, random_state=71)
    X_train = None
    X_val = None
    for tr_group_idx, va_group_idx in kf.split(unique_race_ids):
        tr_groups, va_groups = (
            unique_race_ids[tr_group_idx],
            unique_race_ids[va_group_idx],
        )
        is_tr, is_va = race_id.isin(tr_groups), race_id.isin(va_groups)
        X_train, X_val = dataset[is_tr], dataset[is_va]
        break

    return X_train, X_val


def calibration(y_proba, beta):
    return y_proba / (y_proba + (1 - y_proba) / beta)


class SmoothingTargetEncodingCV(object):
    """
    クロスバリデーション用のTarget Encording
    データのリークを防ぐために、KFoldで分割してout-of-foldで変換した値に置換する
    """

    def __init__(self, columns, k=100):
        self.columns = columns
        self.train = None
        self.target = None
        self.val_train = None
        self.target_mean = None
        self.k = k

    def _sigmoid(self, count):
        return 1 / (1 + np.exp(-count / self.k))

    def _smoothing(self, target):
        averages = target.groupby(by="key")["target"].agg(["mean", "count"])
        sig = self._sigmoid(averages["count"])
        return self.target_mean * (1 - sig) + averages["mean"] * sig

    def fit(self, train, target, val_train):
        self.train = train
        self.target = target
        self.val_train = val_train
        self.target_mean = target.mean()

    def transform(self):
        for col in self.columns:
            # バリデーション用データはTrainセットを使ってTarget Encordingする
            data_tmp = pd.DataFrame({"key": self.train[col], "target": self.target})
            smoothing_values = self._smoothing(data_tmp)
            self.val_train.loc[:, col] = self.val_train[col].map(smoothing_values)
            self.val_train[col] = self.val_train[col].astype("float64")

            # 訓練用データはKFoldでout-of-foldを使ってTarget Encordingする
            tmp = np.repeat(np.nan, self.train.shape[0])
            fold = 4
            kf_encoding = KFold(n_splits=fold, shuffle=True, random_state=72)
            for idx_1, idx_2 in kf_encoding.split(self.train):
                # KFoldの訓練データで集計した値を
                t_smoothing_values = self._smoothing(data_tmp.iloc[idx_1])
                # テストデータ部分に置換することでout-of-foldを実現している
                tmp[idx_2] = self.train[col].iloc[idx_2].map(t_smoothing_values)

            self.train.loc[:, col] = tmp

        return self.train, self.val_train


class SmoothingTargetEncoding(object):
    """
    過去のデータを使ったTarget Encording
    """

    def __init__(self, columns, k=100):
        self.columns = columns
        self.from_dataset = None
        self.target = None
        self.to_dataset = None
        self.target_mean = None
        self.aggr_dict = {}
        self.k = k

    def _sigmoid(self, count):
        return 1 / (1 + np.exp(-count / self.k))

    def _smoothing(self, target):
        averages = target.groupby(by="key")["target"].agg(["mean", "count"])
        sig = self._sigmoid(averages["count"])
        return self.target_mean * (1 - sig) + averages["mean"] * sig

    def fit(self, from_dataset, target, to_dataset):
        """
        Parameters
        ----------
        from_dataset : pandas.DataFrame
            集計元となるデータセット
        target : pandas.DataFrame
            集計元となるターゲット
        to_dataset : pandas.DataFrame
            置換したいデータセット
        """
        self.from_dataset = from_dataset
        self.target = target
        self.to_dataset = to_dataset
        self.target_mean = target.mean()
        self.aggr_dict = {}

    def transform(self):
        for col in self.columns:
            # バリデーション用データはTrainセットを使ってTarget Encordingする
            data_tmp = pd.DataFrame(
                {"key": self.from_dataset[col], "target": self.target}
            )
            smoothing_values = self._smoothing(data_tmp)
            self.to_dataset.loc[:, col] = self.to_dataset[col].map(smoothing_values)
            self.to_dataset[col] = self.to_dataset[col].astype("float64")
            # 他の場所で使いまわしたい場合のために返却しておく
            self.aggr_dict[col] = smoothing_values

        return self.to_dataset, self.aggr_dict


class LightGBMInterface(object):
    def target_encoding_cv(self, train, test, target_name, te_columns):
        # Target Encording
        ste = SmoothingTargetEncodingCV(te_columns)
        ste.fit(train, train[target_name], test)
        ste.transform()

        return train, test

    def under_sampler(
        self, X_train, train_label, X_val, category_columns, random_state
    ):
        # アンダーサンプリングのデータ数を決める
        label = train_label.columns[0]
        count_map = train_label.groupby(by=label)[label].count()
        sampling_size = count_map.min()
        sampling_strategy = dict()
        for k in count_map.keys():
            sampling_strategy[k] = sampling_size

        # ダウンサンプリング
        # Nullが含まれているとエラーになるのでいったん置換してあとで戻す
        X_train_n = X_train.fillna(-100000)
        train_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
        X_train_sampled, train_label_sampled = train_sampler.fit_sample(
            X_train_n, train_label
        )
        X_train_sampled = pd.DataFrame(X_train_sampled, columns=X_train_n.columns)
        train_label_sampled = pd.DataFrame(
            train_label_sampled, columns=train_label.columns
        )
        X_train_sampled = X_train_sampled.replace(-100000, pd.np.nan)

        X_train_sampled[category_columns] = X_train_sampled[category_columns].astype(
            "category"
        )
        X_val[category_columns] = X_val[category_columns].astype("category")

        return X_train_sampled, train_label_sampled, X_val

    def _split_train_label(self, dataset, label_cols):
        label = pd.DataFrame(dataset[label_cols])
        train = dataset.drop([label_cols], axis=1)
        return train, label


class LightGBMTuner(LightGBMInterface):
    def __init__(
        self,
        parameter,
        meta_cols=[],
        category_columns=[],
        target_encoding_columns=[],
        target_encoding=False,
        under_sampling=False,
    ):
        self.paramter = parameter
        self.category_columns = category_columns
        self.target_encoding_columns = target_encoding_columns
        self.meta_cols = meta_cols
        self.evaluation_results = dict()
        target_encoding = target_encoding
        self.under_sampling = under_sampling

    def train(self, fold, dataset, label_name):
        race_id = dataset.race_id
        unique_race_ids = race_id.unique()
        kf = KFold(n_splits=fold, shuffle=True, random_state=71)
        X_train = None
        X_val = None
        for tr_group_idx, va_group_idx in kf.split(unique_race_ids):
            tr_groups, va_groups = (
                unique_race_ids[tr_group_idx],
                unique_race_ids[va_group_idx],
            )
            is_tr, is_va = race_id.isin(tr_groups), race_id.isin(va_groups)
            X_train, X_val = dataset[is_tr], dataset[is_va]
            break

        if self.target_encoding:
            for h in self.target_encoding_columns:
                X_train, X_val = self.target_encoding_cv(
                    X_train, X_val, h["target"], h["columns"]
                )

        X_train, train_label = self._split_train_label(X_train, label_name)
        X_val, val_label = self._split_train_label(X_val, label_name)

        if self.under_sampling:
            X_train, train_label, X_val = self.under_sampler(
                X_train, train_label, X_val, self.category_columns, 123
            )
        else:
            X_train[self.category_columns] = X_train[self.category_columns].astype(
                "category"
            )
            X_val[self.category_columns] = X_val[self.category_columns].astype(
                "category"
            )

        X_train = X_train.drop(self.meta_cols, axis=1, errors="ignore")
        X_val = X_val.drop(self.meta_cols, axis=1, errors="ignore")

        lgb_train = lgb.Dataset(X_train, train_label)
        lgb_valid = lgb.Dataset(X_val, val_label, reference=lgb_train)

        lgbm = tuner.train(
            self.paramter,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            early_stopping_rounds=100,
            verbose_eval=0,
            evals_result=self.evaluation_results,
            valid_names=["Train", "Test"],
        )

        return lgbm, lgbm.params


class LightGBMCV(LightGBMInterface):
    def __init__(
        self,
        parameter,
        meta_cols=[],
        category_columns=[],
        target_encoding_columns=[],
        target_encoding=False,
        under_sampling=False,
    ):
        self.paramter = parameter
        self.category_columns = category_columns
        self.target_encoding_columns = target_encoding_columns
        self.meta_cols = meta_cols
        self.evaluation_results = dict()
        self.scores = []
        self.lgbm_models = []
        self.target_encoding = target_encoding
        self.under_sampling = under_sampling

    def cv(self, fold, dataset, label_name):
        preds_df = pd.DataFrame()
        race_id = dataset.race_id
        unique_race_ids = race_id.unique()
        kf = KFold(n_splits=fold, shuffle=True, random_state=71)
        cur_idx = 1
        label_types = np.sort(dataset[label_name].unique())

        for tr_group_idx, va_group_idx in kf.split(unique_race_ids):
            tr_groups, va_groups = (
                unique_race_ids[tr_group_idx],
                unique_race_ids[va_group_idx],
            )
            is_tr, is_va = race_id.isin(tr_groups), race_id.isin(va_groups)
            X_train, X_val = dataset[is_tr], dataset[is_va]

            if self.target_encoding:
                for h in self.target_encoding_columns:
                    X_train, X_val = self.target_encoding_cv(
                        X_train, X_val, h["target"], h["columns"]
                    )

            X_train, train_label = self._split_train_label(X_train, label_name)
            X_val, val_label = self._split_train_label(X_val, label_name)

            if self.under_sampling:
                X_train, train_label, X_val = self.under_sampler(
                    X_train, train_label, X_val, self.category_columns, (cur_idx * 128)
                )
            else:
                X_train[self.category_columns] = X_train[self.category_columns].astype(
                    "category"
                )
                X_val[self.category_columns] = X_val[self.category_columns].astype(
                    "category"
                )

            X_train = X_train.drop(self.meta_cols, axis=1, errors="ignore")
            X_val_col = X_val[["race_id", "horse_id"]].copy().reset_index(drop=True)
            X_val = X_val.drop(self.meta_cols, axis=1, errors="ignore")

            lgbm = self.train(X_train, train_label, X_val, val_label)
            va_pred = lgbm.predict(X_val)
            self.lgbm_models.append(lgbm)

            val_preds_df = self.get_validation_result(val_label, va_pred, label_types)

            tmp = pd.concat(
                [X_val_col, val_preds_df, val_label.reset_index(drop=True)], axis=1
            )
            preds_df = preds_df.append(tmp, ignore_index=True, sort=False)

            cur_idx += 1

        return preds_df

    def train(self, train, train_label, test, test_label):
        lgb_train = lgb.Dataset(train, train_label)
        lgb_valid = lgb.Dataset(test, test_label, reference=lgb_train)

        lgbm = lgb.train(
            self.paramter,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            early_stopping_rounds=100,
            verbose_eval=0,
            evals_result=self.evaluation_results,
            valid_names=["Train", "Test"],
        )

        return lgbm

    def get_validation_result(self, val_label, va_pred, label_types):
        val_preds_df = None
        if self.paramter["metric"] == "rmse":
            score = mean_squared_error(val_label, va_pred)
            self.scores.append(score)
            val_preds_df = pd.DataFrame(va_pred, columns=["predict"])
        if self.paramter["metric"] == "mae":
            score = mean_absolute_error(val_label, va_pred)
            self.scores.append(score)
            val_preds_df = pd.DataFrame(va_pred, columns=["predict"])
        elif self.paramter["metric"] == "multi_logloss":
            score = log_loss(val_label, va_pred)
            self.scores.append(score)
            pred_columns = []
            for lt in label_types:
                pred_columns.append("class{}".format(lt))
            val_preds_df = pd.DataFrame(va_pred, columns=pred_columns)
        elif self.paramter["metric"] == "binary_logloss":
            score = log_loss(val_label, va_pred)
            self.scores.append(score)
            val_preds_df = pd.DataFrame(va_pred, columns=["predict"])

        return val_preds_df

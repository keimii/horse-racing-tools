import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold


def calibration(y_proba, beta):
    return y_proba / (y_proba + (1 - y_proba) / beta)


class LightGBMCV(object):
    def __init__(self, parameter, meta_cols, category_columns, under_sampling=False):
        self.paramter = parameter
        self.category_columns = category_columns
        self.meta_cols = meta_cols
        self.evaluation_results = dict()
        self.scores = []
        self.lgbm_models = []
        self.under_sampling = under_sampling

    def _split_train_label(self, dataset, label_cols):
        label = pd.DataFrame(dataset[label_cols])
        train = dataset.drop([label_cols], axis=1)
        return train, label

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

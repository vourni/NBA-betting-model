from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
import numpy as np

class PurgedGroupTimeSeriesSplit:
    """Drop-in replacement for TimeSeriesSplit with group-aware purging."""
    def __init__(self, n_splits=5, *, group_gap=1, max_train_group_size=None, test_group_size=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.group_gap = int(group_gap)
        self.max_train_group_size = max_train_group_size
        self.test_group_size = test_group_size

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("PurgedGroupTimeSeriesSplit requires 'groups' (e.g. normalized dates)")
        groups = np.asarray(groups)
        unique, first_idx = np.unique(groups, return_index=True)
        order = np.argsort(first_idx)
        unique_groups = unique[order]
        group_to_pos = {g: i for i, g in enumerate(unique_groups)}
        group_ids = np.fromiter((group_to_pos[g] for g in groups), dtype=int, count=len(groups))
        n_groups = len(unique_groups)
        if self.n_splits >= n_groups:
            raise ValueError("Too few unique groups for the requested number of splits")

        test_size = self.test_group_size or max(1, n_groups // (self.n_splits + 1))

        for fold in range(self.n_splits):
            test_start = n_groups - (self.n_splits - fold) * test_size
            test_end = min(test_start + test_size, n_groups)
            if test_start <= 0:
                continue
            train_end = max(0, test_start - self.group_gap)
            if self.max_train_group_size is not None:
                train_start = max(0, train_end - int(self.max_train_group_size))
            else:
                train_start = 0

            train_mask = (group_ids >= train_start) & (group_ids < train_end)
            val_mask = (group_ids >= test_start) & (group_ids < test_end)
            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class TwoStageStackTS(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, base_estimators, meta_estimator, n_splits=6, gap=2):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.n_splits = n_splits
        self.gap = gap

    def _safe_slice(self, X, idx):
        return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    def fit(self, X, y, groups):
        y = np.asarray(y).ravel()
        splitter = PurgedGroupTimeSeriesSplit(n_splits=self.n_splits, group_gap=self.gap)
        Zcols, self.base_fitted_ = [], []

        for name, est in self.base_estimators:
            oof = np.full(len(y), np.nan)
            for tr, te in splitter.split(X, y, groups=groups):
                e = clone(est)
                e.fit(self._safe_slice(X, tr), y[tr])
                oof[te] = e.predict_proba(self._safe_slice(X, te))[:, 1]
            Zcols.append(oof)
            self.base_fitted_.append((name, clone(est).fit(X, y)))
        Z = np.column_stack(Zcols)
        mask = np.all(~np.isnan(Z), axis=1)
        self.meta_ = clone(self.meta_estimator).fit(Z[mask], y[mask])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["meta_", "base_fitted_"])
        cols = [est.predict_proba(X)[:, 1] for _, est in self.base_fitted_]
        return self.meta_.predict_proba(np.column_stack(cols))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
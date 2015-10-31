"""Module for different Decision Tree implementations"""

import numpy as np

from ..utils import majority_val, series_info


class NaiveNonNumericDecisionTree(object):
    """Decision tree with no pruning, missing values, or numerical attributes"""

    def __init__(self):
        self._top_node = None
        self._default_val = None
        self._predict_field = None
        self._predict_default = None
        self._count_field = "_counts"

    @staticmethod
    def _create_decision_node(attr, values):
        return {'attr': attr, 'values': {value: None for value in values}}

    def _create_decision_tree(self, df, attrs=None, default=None):
        if len(df) == 0:
            return default
        if len(df[self._predict_field].unique()) <= 1:
            return df[self._predict_field].iloc[0]
        if attrs is None:
            attrs = list(set(df.columns) - {self._predict_field, self._count_field})
        if len(attrs) == 0:
            return majority_val(df, self._predict_field)
        best_attr = self._choose_attr(df, attrs)
        best_attr_vals = df[best_attr].unique()
        tree = self._create_decision_node(best_attr, best_attr_vals)
        maj_vals = majority_val(df, best_attr)
        new_attrs = [val for val in attrs if val != best_attr]
        for val in best_attr_vals:
            df_v = df[df[best_attr] == val]
            sub_tree = self._create_decision_tree(df_v, new_attrs, maj_vals)
            tree['values'][val] = sub_tree
        return tree

    def _choose_attr(self, df, attrs):
        gains = {attr: self._attr_gain(df, attr) for attr in attrs}
        return max(gains, key=lambda x: gains[x])

    def _attr_gain(self, df, attr):
        s = df[self._predict_field]
        attr_total_info = series_info(s)
        attr_remainders = self._groupby_remainder(df, attr)
        return attr_total_info - sum(attr_remainders.values())

    def _groupby_remainder(self, df, gb_key):
        attr_counts = df.groupby(gb_key)[self._count_field].sum()
        attr_pred_counts = df.groupby([gb_key, self._predict_field])[self._count_field].sum()
        attr_names = attr_pred_counts.index.levels[0]
        attr_info = {attr_name: series_info(attr_pred_counts[attr_name]) for attr_name in attr_names}
        attr_remainders = {attr_name: attr_counts[attr_name] / len(df) * attr_info[attr_name] for attr_name in
                           attr_names}
        return attr_remainders

    def fit(self, df, predict_field="class", default_val=None):
        df[self._count_field] = np.ones(len(df))
        self._predict_field = predict_field
        self._predict_default = default_val if default_val else majority_val(df, predict_field)
        self._top_node = self._create_decision_tree(df)
        return True

    def predict(self, x):
        curr_node = self._top_node
        while isinstance(curr_node, dict):
            attr, values = curr_node["attr"], curr_node["values"]
            curr_node = values.get(x[attr], self._predict_default)
        return curr_node


class NaiveNumericDecisionTree(NaiveNonNumericDecisionTree):
    """Decision tree with no pruning or missing values but with numerical attributes"""

    def _attr_gain(self, df, attr):
        use_continuous_gain = False
        if np.issubdtype(df.dtypes[attr], np.number):
            splits = df[attr].unique()
            if len(splits) > 5:  # Should really have a way to check if data is ordinal versus continuous.
                use_continuous_gain = True
        if use_continuous_gain:
            return self._attr_gain_continuous(df, attr)
        else:
            return super(self, NaiveNumericDecisionTree)._attr_gain(df, attr), None

    def _attr_gain_continuous(self, df, attr):
        splits = list(df[attr].unique())
        splits.remove(min(splits))
        attr_total_info = series_info(df[self._predict_field])
        split_gain = {}
        for split in splits:
            attr_remainders = self._groupby_remainder(df, lambda idx: df[attr][idx] < split)
            split_gain[split] = attr_total_info - sum(attr_remainders.values())
        max_gain = max(split_gain.values())
        split_max_gain = filter(lambda k: split_gain[k] == max_gain, split_gain).__next__()
        return max_gain, split_max_gain

    def _choose_attr(self, df, attrs):
        gains = {attr: self._attr_gain(df, attr) for attr in attrs}
        return max(gains, key=lambda x: gains[x][0])

    def predict(self, x):
        pass

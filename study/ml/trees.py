"""Module for different Decision Tree implementations"""

import math

import numpy as np

from ..utils import majority_val


class NaiveNonNumericDecisionTree(object):
    """Decision tree with no pruning or numerical attributes"""

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
        attr_total_info = self._information(s)

        attr_counts = df.groupby(attr)[self._count_field].sum()
        attr_pred_counts = df.groupby([attr, self._predict_field])[self._count_field].sum()
        attr_names = attr_pred_counts.index.levels[0]
        attr_info = {attr_name: self._information(attr_pred_counts[attr_name]) for attr_name in attr_names}
        attr_remainders = {attr_name: attr_counts[attr_name] / len(df) * attr_info[attr_name] for attr_name in
                           attr_names}
        return attr_total_info - sum(attr_remainders.values())

    @staticmethod
    def _information(s):
        return sum([-p * math.log2(p) for p in s.groupby(lambda x: s[x]).count() / len(s)])

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

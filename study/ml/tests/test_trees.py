import unittest

import pandas as pd
from study.ml.trees import NaiveNumericDecisionTree

from test_classifiers import ClassifierBaseTest


class TestNonNumericDecisionTree(ClassifierBaseTest):
    data2_df = pd.DataFrame({'one': [1., 2., 3., 4., 5., 6.],
                             'two': [True, True, False, False, False, False],
                             "_counts": [1, 1, 1, 1, 1, 1]
                             })

    def test_attr_gain_continuous(self):
        """Test splitting numerics attributes"""
        dtree = NaiveNumericDecisionTree()
        dtree._predict_field = 'two'
        gain = dtree._attr_gain_continuous(self.data2_df, 'one')
        self.assertEqual(gain[1], 3.0)


if __name__ == '__main__':
    unittest.main()

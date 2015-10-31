import unittest

import pandas as pd

from study.ml.boosters import AdaBoost
from study.ml.classifiers import DecisionTreeClassifier
from test_classifiers import ClassifierBaseTest


class TestAdaBoost(ClassifierBaseTest):
    """Test the AdaBooster"""

    def test_creation(self):
        """Test creation of AdaBooster"""
        classifier = AdaBoost()
        self.assertTrue(classifier)

    def test_fit(self):
        """Test fit method of AdaBooster"""
        fit_res = AdaBoost().fit(self.data1_df, "two", DecisionTreeClassifier, 3)
        self.assertTrue(fit_res)

    @unittest.skip("prediction not implemented")
    def test_prediction(self):
        """Test predict method of AdaBooster"""
        classifier = AdaBoost()
        predict_field = "two"
        classifier.fit(self.data1_df, predict_field)
        predict_input = pd.Series([1., ], index=["one", ])
        prediction = classifier.predict(predict_input)
        self.assertEqual(prediction, self.data1_df[predict_field].mode()[0])


if __name__ == '__main__':
    unittest.main()

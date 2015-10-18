import unittest

import pandas as pd

from study.ml.classifiers import LookUpClassifier, MajorityClassifier


class ClassifierBaseTest(unittest.TestCase):

    data1_df = pd.DataFrame({'one': [1., 2., 3., 4.],
                             'two': [1., 3., 2., 1.]})


class TestLookupClassifier(ClassifierBaseTest):
    """Test the LookupClassifier"""

    def test_creation(self):
        """Test creation of lookup classifier"""
        classifier = LookUpClassifier()
        self.assertTrue(classifier)

    def test_fit(self):
        """Test fit method of lookup classifier"""
        fit_res = LookUpClassifier().fit(self.data1_df, "two")
        self.assertTrue(fit_res)

    def test_prediction_lookup(self):
        """Test predict method of lookup classifier"""
        classifier = LookUpClassifier()
        predict_field = "two"
        classifier.fit(self.data1_df, predict_field)
        predict_input = pd.Series(data=[2., ], index=["one", ])
        prediction = classifier.predict(predict_input)
        self.assertEqual(prediction, self.data1_df[self.data1_df["one"] == 2.][predict_field].iloc[0])

    def test_prediction_default(self):
        """Test predict method of lookup classifier"""
        classifier = LookUpClassifier()
        predict_field = "two"
        classifier.fit(self.data1_df, predict_field)
        predict_input = pd.Series(data=[5., ], index=["one"])
        prediction = classifier.predict(predict_input)
        self.assertEqual(prediction, self.data1_df[predict_field].mode()[0])


class TestMajorityClassifier(ClassifierBaseTest):
    """Test the MajoricyClassifier"""

    def test_creation(self):
        """Test creation of majority classifier"""
        classifier = MajorityClassifier()
        self.assertTrue(classifier)

    def test_fit(self):
        """Test fit method of majority classifier"""
        fit_res = MajorityClassifier().fit(self.data1_df, "two")
        self.assertTrue(fit_res)

    def test_prediction(self):
        """Test predict method of majority classifier"""
        classifier = MajorityClassifier()
        predict_field = "two"
        classifier.fit(self.data1_df, predict_field)
        predict_input = pd.Series([1., ], index=["one", ])
        prediction = classifier.predict(predict_input)
        self.assertEqual(prediction, self.data1_df[predict_field].mode()[0])


if __name__ == '__main__':
    unittest.main()

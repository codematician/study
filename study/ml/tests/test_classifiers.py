import unittest

import pandas as pd

from study.ml.classifiers import MajorityClassifier


class TestMajorityClassifier(unittest.TestCase):
    """Test the MajoricyClassifier"""

    data_df = pd.DataFrame({'one': [1., 2., 3., 4.],
                            'two': [1., 3., 2., 1.]})

    def test_creation(self):
        """Test creation of majority classifier"""
        classifier = MajorityClassifier()
        self.assertTrue(classifier)

    def test_fit(self):
        """Test fit method of majoricy classifier"""
        fit_res = MajorityClassifier().fit(self.data_df, "two")
        self.assertTrue(fit_res)

    def test_prediction(self):
        classifier = MajorityClassifier()
        classifier.fit(self.data_df, "two")
        prediction = classifier.predict([1., ])
        self.assertEqual(prediction, self.data_df['two'].mode()[0])


if __name__ == '__main__':
    unittest.main()

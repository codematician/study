import pandas as pd

from study.ml.boosters import AdaBoost


class TestAdaBoost(ClassifierBaseTest):
    """Test the MajoricyClassifier"""

    def test_creation(self):
        """Test creation of majority classifier"""
        classifier = AdaBoost()
        self.assertTrue(classifier)

    def test_fit(self):
        """Test fit method of majority classifier"""
        fit_res = AdaBoost().fit(self.data1_df, "two")
        self.assertTrue(fit_res)

    def test_prediction(self):
        """Test predict method of majority classifier"""
        classifier = AdaBoost()
        predict_field = "two"
        classifier.fit(self.data1_df, predict_field)
        predict_input = pd.Series([1., ], index=["one", ])
        prediction = classifier.predict(predict_input)
        self.assertEqual(prediction, self.data1_df[predict_field].mode()[0])

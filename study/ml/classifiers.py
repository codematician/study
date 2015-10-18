from abc import ABCMeta, abstractmethod


class Classifier(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, df, predict_field="class"):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class MajorityClassifier(Classifier):
    """A classifer that always give back the majority answer of the training set.

     This classifer is useful for comparing the performance of a bad model.
    """
    def __init__(self):
        self._prediction_value = None

    def fit(self, df, predict_field="class"):
        self._prediction_value = df[predict_field].mode()[0]
        return True

    def predict(self, x):
        return self._prediction_value

from abc import ABCMeta, abstractmethod
import operator
from functools import reduce


class Classifier(metaclass=ABCMeta):
    """A abstrace base class for classifiers.

    Classifiers consist of two functions, fit and predict.

    - fit: The fit method is called with a Pandas Dataframe of labeled training
           data. The prediction field indicates the label that is to be predicted.

    - predict: Takes single row of data and produces the field trained to fit.
    """
    @abstractmethod
    def fit(self, df, predict_field="class"):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class LookUpClassifier(Classifier):
    """A classifier that looks up an answer in a training set.

    Notes
    -----

     This classifier is only useful for comparing the performance of a bad model.

    References
    ----------
    - Russell, Stuart, and Peter Norvig. "Artificial intelligence: a modern approach." (1995). APA

    """
    def __init__(self):
        self._lookup_df = None
        self._predict_field = None
        self._predict_default = None

    def fit(self, df, predict_field="class"):
        self._predict_field = predict_field
        self._predict_default = df[predict_field].mode()[0]
        self._lookup_df = df.drop_duplicates()
        return True

    def predict(self, x):
        criterion = reduce(operator.and_, map(lambda idx: self._lookup_df[idx] == x[idx], x.index))
        lookup = self._lookup_df[criterion]
        predict_val = self._predict_default if len(lookup) == 0 else lookup[self._predict_field].iloc[0]
        return predict_val


class MajorityClassifier(Classifier):
    """A classifier that always give back the majority answer of the training set.

    Notes
    -----

     This classifier is only useful for comparing the performance of a bad model.

    """
    def __init__(self):
        self._prediction_value = None

    def fit(self, df, predict_field="class"):
        self._prediction_value = df[predict_field].mode()[0]
        return True

    def predict(self, x):
        return self._prediction_value

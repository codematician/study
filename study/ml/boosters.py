"""Boosting algorithms"""


class AdaBoost(object):
    """AdaBoost algorithm for boosting an ensemble of learners.

    Notes
    -----


    References
    ----------
      - Russell, Stuart, and Peter Norvig. "Artificial intelligence: a modern approach." (1995). APA

    """
    def __init__(self):
        self._predict_field = None
        self._num_learners = None
        self._learners = None
        self._learners_weights = None

    def fit(self, examples, predict_field, learning_alg, num_learners):
        self._examples = examples
        self._predict_field = predict_field
        self._learning_alg = learning_alg
        self._num_learners = num_learners
        self._learners = [None] * self._num_learners
        self._learners_weights = [1.0/self._num_learners] * self._num_learners
        return True

    def predict(self, x):
        raise NotImplementedError

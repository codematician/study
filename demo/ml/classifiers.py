"""Example demonstrating calling classifiers"""

import random

import pandas as pd

from study.datasets import adult_df, cars_df
from study.ml.classifiers import DecisionTreeClassifier, LookUpClassifier, MajorityClassifier
from study.ml.trees import NaiveNumericDecisionTree

CLASSIFIERS = {"decision": DecisionTreeClassifier,
               "numeric_decision": lambda: DecisionTreeClassifier(tree_class=NaiveNumericDecisionTree),
               "lookup": LookUpClassifier,
               "majority": MajorityClassifier,
               }


def run(df, classifiers, num_train=500):
    classifiers = {k: v() for k, v in classifiers.items()}
    rows = random.sample(list(df.index), num_train)
    train = df.ix[rows]
    test = df.drop(rows)
    list(map(lambda c: c.fit(train), classifiers.values()))
    pred_succ = {name: sum(c.predict(row[1]) == row[1].ix["class"] for row in test.iterrows()) / len(test)
                 for name, c in classifiers.items()}
    return pred_succ


def main():
    preds_df = pd.DataFrame([], columns=CLASSIFIERS.keys())
    for num_train in [10 * 2 ** i for i in range(1, 8)]:
        print("Running cars training set size: {0}".format(num_train))
        pred_dict = run(cars_df, CLASSIFIERS, num_train)
        preds_df = preds_df.append(pd.Series(pred_dict, name=num_train))
    print("Cars Prediction rates")
    print(preds_df)

    numeric_classifiers = ["numeric_decision"]
    preds_df = pd.DataFrame([], columns=numeric_classifiers)
    for num_train in [10 * 2 ** i for i in range(1, 8)]:
        print("Running adult training set size: {0}".format(num_train))
        pred_dict = run(adult_df, {k: CLASSIFIERS[k] for k in numeric_classifiers}, num_train)
        preds_df = preds_df.append(pd.Series(pred_dict, name=num_train))
    print("Adult Prediction rates")
    print(preds_df)


if __name__ == "__main__":
    main()

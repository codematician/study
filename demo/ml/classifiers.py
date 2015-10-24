"""Example demonstrating calling classifiers"""

import random

import pandas as pd

from study.datasets import cars_df
from study.ml.classifiers import DecisionTreeClassifier, LookUpClassifier, MajorityClassifier

CLASSIFIERS = {"decision": DecisionTreeClassifier,
               "lookup": LookUpClassifier,
               "majority": MajorityClassifier,
               }


def run(num_train=500):
    classifiers = {k: v() for k, v in CLASSIFIERS.items()}
    rows = random.sample(list(cars_df.index), num_train)
    train = cars_df.ix[rows]
    test = cars_df.drop(rows)
    list(map(lambda c: c.fit(train), classifiers.values()))
    pred_succ = {name: sum(c.predict(row[1]) == row[1].ix["class"] for row in test.iterrows()) / len(test)
                 for name, c in classifiers.items()}
    return pred_succ


def main():
    preds_df = pd.DataFrame([], columns=CLASSIFIERS.keys())
    for num_train in [10*2**i for i in range(1, 8)]:
        print("Running training set size: {0}".format(num_train))
        pred_dict = run(num_train)
        preds_df = preds_df.append(pd.Series(pred_dict, name=num_train))
    print(preds_df)


if __name__ == "__main__":
    main()

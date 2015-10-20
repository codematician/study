"""Example demonstrating calling classifiers"""

from pprint import pprint

from study.datasets import cars_df
from study.ml.classifiers import LookUpClassifier, MajorityClassifier


def main(num_test=10):
    classifiers = {"lookup": LookUpClassifier(), "majority": MajorityClassifier()}
    train, test = cars_df[:-num_test], cars_df[-num_test:]
    list(map(lambda c: c.fit(train), classifiers.values()))
    predictions = {name: sum(c.predict(row[1]) == row[1].ix["class"] for row in test.iterrows()) / num_test for name, c in
                   classifiers.items()}
    print("Classification success percent")
    print(predictions)


if __name__ == "__main__":
    main()

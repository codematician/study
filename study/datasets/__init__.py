import os

import pandas as pd


cars_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "car", "car.data"),
                      names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
                      index_col=False)

adult_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "adult", "adult.data"),
                       names=["age", "workclass","fnlwgt", "education", "education-num", "marital-status"," occupation",
                              "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                              "native-country", "class"],
                       index_col=False)

import os

import pandas as pd

print(os.path.dirname(__file__))

cars_df = pd.read_csv(os.path.dirname(__file__)+"/car/car.data",
                      names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
                      index_col=False)

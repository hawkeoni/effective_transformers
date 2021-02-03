import random
from pathlib import Path

import pandas as pd


random.seed(42)
df_dict = {"Source": [], "Target": []}
for i in range(100):
    source = [random.randint(0, 9) for _ in range(3)]
    target = sum(source) % 10
    df_dict["Source"].append(" ".join(map(str, source)))
    df_dict["Target"].append(target)
df = pd.DataFrame(df_dict)
Path("dataset").mkdir(exist_ok=True)
df.to_csv("dataset/basic_train.csv", index=False)


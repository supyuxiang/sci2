import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

from sympy.core.facts import FactRules

current_path = Path(__file__)

data_path = current_path.parent / 'data' / '三维双热通量.xlsx'  # 三维双热通量.xlsx 是绝对路径

data = pd.read_excel(data_path)

def find_index(data:pd.DataFrame):
    idx = 10
    play = True
    while play:
        if pd.isnull(data.iloc[idx,6]):
            play = not play
        idx += 1
    return idx

idx_end = find_index(data)



idx = 8
print(data.iloc[idx,:])

print('*'*100)

for i,x in enumerate(data.iloc[0:15,0]):
    print(i,x)

print('*'*100)





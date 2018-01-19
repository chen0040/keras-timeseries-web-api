import pandas as pd
import numpy as np
import os

data_dir_path = '../data'

frame = pd.read_csv(filepath_or_buffer=os.path.join(data_dir_path, 'monthly-milk-production-pounds-p.csv'), sep=',')

print(frame.as_matrix(["MilkProduction"]).T)

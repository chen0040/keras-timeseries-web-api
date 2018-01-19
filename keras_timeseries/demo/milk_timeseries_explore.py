import pandas as pd
import numpy as np
import os

DATA_DIR = './data'
FILE_NAME = 'monthly-milk-production-pounds-p.csv'

frame = pd.read_csv(filepath_or_buffer=os.path.join(DATA_DIR, FILE_NAME), sep=',')

print(frame.as_matrix(["MilkProduction"]).T)

import pandas as pd
import matplotlib.pyplot as plt

df0 = pd.read_csv('results0')
df1 = pd.read_csv('results1')

df0.groupby(['n']).sum()
df1.groupby(['n']).sum()

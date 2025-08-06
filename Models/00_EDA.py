import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew


df=pd.read_csv('cleaned_survey.csv')

plt.figure(figsize=(10,12))
sns.histplot(data=np.log1p(df['Age']),kde=True)
plt.show()

print("Original Skew:", skew(df['Age']))
print("Log1p Skew:", skew(np.log1p(df['Age'])))
print(df.isnull().sum())
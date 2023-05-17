import pandas as pd
import numpy as np

df = pd.DataFrame(columns=['awG_kind','awL_kind','awG','awL','e_RMS'])
print(df)

##new_row = {'awG_kind':'linear','awL_kind':'linear','awG':0.3,'awL':0.4,'e_RMS':0.02}
new_row = ['linear','linear',0.3,0.4,0.02]
df.loc[df.shape[0]] = new_row
df.loc[df.shape[0]] = new_row
df.loc[df.shape[0]] = new_row
##df = df.append(new_row,ignore_index=True)
print(df)

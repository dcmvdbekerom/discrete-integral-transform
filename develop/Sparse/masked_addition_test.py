
import numpy as np




arr = np.arange(20)

index_arr = np.array([3,4,7,8,10])
val_arr = np.array([100,100,100,100,100])

np.add.at(arr,index_arr,val_arr)
print(arr)

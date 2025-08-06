import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
d1=np.array([[[1,1],[0.9,0.9]],[[0,0],[0,0]],[[0,0],[1,1]],[[0,0],[0,0]],
             [[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0.9,0.9],[1,1]]])
norm=Normalize(vmin=0,vmax=1)
fig,ax=plt.subplots(2,4)
a=[]
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(d1[i],norm=norm)
    plt.axis('off')

plt.show()

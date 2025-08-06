import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

'''
这个文件是为了画图
'''
def plot_map(arr,title):
    # 将1维数组转化为5x5二维数组
    arr_2d = np.reshape(arr, (5, 5))

    # 使用seaborn绘制热力图
    fig=plt.figure(figsize=(6, 6))
    cmap = ListedColormap(['white', 'black'])
    sns.heatmap(arr_2d, annot=False,cmap=cmap,cbar=False, linewidths=2.5,
                linecolor='black',xticklabels=False, yticklabels=False)
    # plt.show()
    fig.savefig('plot_map/'+title+'.png')
    plt.close()

def main():
    path='hybirdtrainset/hybirdtrainset_0.xlsx'
    img=np.delete(np.array(pd.read_excel(path)).T,0,0)
    for i in range(10):
        plot_map(img[i],'Ethylene%d'%i)
        plot_map(img[i+10],'Ethanol%d'%i)
        plot_map(img[i+20],'Carbon Monoxide%d'%i)
        plot_map(img[i+30],'Methane%d'%i)

if __name__ == '__main__':
    main()
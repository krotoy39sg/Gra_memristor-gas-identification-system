import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.colors import ListedColormap

# 基本参数
device_num = 16
sample_num = 160  # 可以根据具体数据结构确定
odour_num = 4
A = 2.4e12
T = 300
FYB = 0.9
k = 1.38e-23
e0 = 1.6e-19
ebsulu_0 = 8.85e-12
ebsulu_r = 90
Ron = 0.0003
d = 40e-9
a = k * T / e0
b = e0 / (4 * math.pi * ebsulu_0 * ebsulu_r * d)
c = A * T * T


class rc():
    def __init__(self, device_num=320):
        self.device_num = device_num
        self.para = np.zeros((device_num, 4))
        self.para[:, 0] = 0.7166  # 这数据都动不得，参考王童论文表3-2
        self.para[:, 1] = 0.6195
        self.para[:, 2] = 0.04055
        self.para[:, 3] = -0.04
        self.current = np.random.random((device_num, 5)) * 50 + 300
        self.flag = np.zeros(device_num)
        self.v_read = 0.9
        self.w_see = np.zeros(1200)
        self.virtual_node_num = 25#虚拟结点25个
        self.virtual_node_time = 200#虚拟结点时间1s

    def renew_i(self, v):
        # 根据i解w
        t2 = a * np.log(self.current[:, -1] / c) + FYB
        t4 = np.power((a * np.log(self.current[:, -1] / c) + FYB), 2) / b  # t4=(V-EMF)/(1-w)
        w = (t4 - 0.6) / (t4 - self.current[:, -1] * Ron)
        # w = 1 - b / np.power((a * np.log(c * self.current[:, -1]) + FYB), 2)
        dw = (v * self.para[:, 0] * np.power(np.power((1 - w), 2), self.para[:, 1])
              + (1 - v) * (self.para[:, 2] * w + self.para[:, 3]))
        w += dw
        w[w < 0.1] = 0.1
        w[w > 1] = 1
        for i in range(15):
            dw = (self.para[:, 2] * w + self.para[:, 3])
            w += dw
            for j in range(self.device_num):
                if self.flag[j] == 1:
                    if (w[j] < 0.8): w[j] = 0.8
                else:
                    if (w[j] < 0.1): w[j] = 0.1
            w[w > 0.9999] = 0.9999
            if i % 3 == 2:
                v1 = self.v_read * np.ones(self.device_num)
                I1 = np.zeros(self.device_num)
                for j in range(10):
                    f1 = np.power((b * (v1 - 0.3) / (1 - w)), 0.5)
                    I1 = c * np.exp((f1 - FYB) / a)
                    v2 = I1 * Ron * w
                    dev_v = v1 + v2 - self.v_read
                    v1 -= dev_v / 5
                    v1[v1 < 0.301] = 0.301
                self.current[:, i // 3] = I1

    def data_saver(self, file, path):
        # 将数据保存进excel
        data = pd.DataFrame(file)
        writer = pd.ExcelWriter(path)
        data.to_excel(writer)
        writer.close()

    def draw_data(self, data, ss):#画图
        plt.figure(figsize=(10,5))
        x = np.linspace(1, 100, data.shape[1])#生成间隔相同的数列
        y = np.linspace(1, 100, ss.shape[1])
        for i in range(data.shape[0]):
            # plt.subplot(2, 4, i+1)
            plt.plot(x, data[i] * 1e6, y, ss[i])
            # plt.ylim(0, 3)
        plt.show()

    def run_rc(self, spikes, path):
        output_j = np.zeros((odour_num, sample_num, device_num, 505))
        for i in range(odour_num):
            for j in range(sample_num):
                print(i, j)
                self.current = np.random.random((device_num, 5)) * 50 + 300
                for m in range(100):
                    self.renew_i(spikes[i, j, :, m])
                    output_j[i, j, :, (m + 1) * 5: (m + 2) * 5] = self.current * (math.pi * np.power((5e-5), 2) / 4)
                if np.sum(spikes[i, j]) > 1:
                    ss = np.zeros((device_num, 1000))
                    for m in range(100):
                        for n in range(device_num):
                            if spikes[i, j, n, m] == 1:
                                ss[n, 10 * m:10 * m + 5] = 0.1
                    # self.draw_data(output_j[i, j], ss)
        save_data = np.zeros((odour_num, sample_num, device_num, 20))
        for i in range(20):
            save_data[:, :, :, i] = output_j[:, :, :, (i + 1) * 25] * 1e6
        # self.draw_fig(save_data)
        # self.data_saver(save_data.reshape((odour_num * sample_num, device_num * 20)), path)

    def new_run_rc(self, spikes,path):#主要用这个当RC
        output_j = np.zeros((spikes.shape[0], spikes.shape[1] * 5 + 5))  # 写脉冲1.5ms，3V，每一个写脉冲之后施加5个读脉冲
        output_j[:, 0:5] = self.current * (math.pi * np.power((5e-5), 2) / 4) * (
                0.9 + 0.2 * np.random.random(self.current.shape))  # 电流
        for m in range(spikes.shape[1]):  # m为spikes的时间长度
            for n in range(spikes.shape[0]):  # n为样本数量
                if spikes[n, m] > 0 and self.flag[n] == 0:
                    self.flag[n] = 1
            self.renew_i(spikes[:, m])
            output_j[:, (m + 1) * 5: (m + 2) * 5] = self.current * (math.pi * np.power((5e-5), 2) / 4) * (
                    0.9 + 0.2 * np.random.random(self.current.shape))
        s_s = np.zeros((self.device_num, output_j.shape[1] * 10))
        for m in range(spikes.shape[1]):
            for n in range(self.device_num):
                if spikes[n, m] == 1:
                    s_s[n, 10 * (m + 1) * 5: 10 * 5 * (m + 1) + 5] = 3
        # self.data_saver(output_j * 1e6, path=r'E:/新建文件夹/工作/SSI/gasdata/Project/RC_data/test_data/speed_output.xlsx')
        # self.data_saver(s_s, path=r'E:/新建文件夹/工作/SSI/gasdata/Project/RC_data/test_data/speed_spike.xlsx')
        # self.draw_data(output_j, s_s * 0.1)
        self.data_saver(output_j, path)  # path=r'C:\Users\czy\Desktop\气敏传感器\画图\matrix_output.xlsx')
        # self.draw(output_j, s_s * 0.1)


    def Virtual_node(self, input_path,output_path):
        output = np.array(pd.read_excel(input_path))

        output = np.delete(output, 0, axis=1)
        self.virtual_node_time = int(output.shape[1] // self.virtual_node_num)
        feature = np.zeros((self.device_num, self.virtual_node_num))
        for i in range(self.virtual_node_num):
            feature[:, i] = output[:, i * self.virtual_node_time]
        # feature= self.min_max_normalize(feature)
            # feature[:, i] = output[:, i * int(5 * self.virtual_node_num]

        self.data_saver(feature.T, output_path)#r'C:\Users\czy\Desktop\气敏传感器\画图\matrix_compress.xlsx')
        # return feature

def re_test():
    spikes = np.zeros((1, 50))
    spikes[0, 1:29] = 1
    rc1 = rc(device_num=1)
    rc1.new_run_rc(spikes,path='E:/新建文件夹/工作/SSI/gasdata/Project/RC_data/test_data/test.xlsx')

def data_saver2(file, path):
     # 将数据保存进excel
    data = pd.DataFrame(file)
    writer = pd.ExcelWriter(path)
    data.to_excel(writer)
    writer.close()

def min_max_normalize(data):
    min_val=np.min(data)
    max_val=np.max(data)
    normalized_data=(data-min_val)/(max_val-min_val)
    return normalized_data

def trainset(datapath,savepath1,savepath2):#将TGS2611和TGS2602的数据集提取出来
    data=np.array(pd.read_excel(datapath))
    data=np.delete(data,0,1)
    data1=np.zeros((40,25))#TGS2611
    data2=np.zeros((40,25))#TGS2602
    for i in range(40):
        data1[i]=data[:,i]
        data2[i]=data[:,i+120]
    data1=min_max_normalize(data1)#正则化之后钯》0.3的数变为1，其余变为0
    data2=min_max_normalize(data2)
    data1[data1>=0.3]=1
    data1[data1<0.3]=0
    data2[data2>=0.3]=1
    data2[data2<0.3]=0
    data_saver2(data1.T,savepath1)
    data_saver2(data2.T,savepath2)

def hybirdtrainset(datapath,savepath):#仅执行正则化
    data = np.array(pd.read_excel(datapath))
    data1 = np.delete(data, 0, 1)
    data1=min_max_normalize(data1)
    data1[data1 >= 0.3] = 1
    data1[data1 < 0.3] = 0
    data_saver2(data1,savepath)


def plot_map(arr,title,i):
    # 将1维数组转化为5x5二维数组
    arr_2d = np.reshape(arr, (5, 5))

    # 使用seaborn绘制热力图
    fig=plt.figure(figsize=(6, 6))
    cmap = ListedColormap(['white', 'black'])
    sns.heatmap(arr_2d, annot=False,cmap=cmap,cbar=False, linewidths=1,
                linecolor='black',xticklabels=False, yticklabels=False)
    plt.title(title)
    # plt.show()
    fig.savefig('E:/新建文件夹/工作/SSI/gasdata/Project/hybirdtrainset/examplemap/'+title+'_%d'%i)
    plt.close()

def union(path1,path2,path3):#这个函数会把两个气敏器件响应结果并集的部分提取出来，混合合并法
    data1=np.delete(np.array(pd.read_excel(path1)),0,1)#611
    data2=np.delete(np.array(pd.read_excel(path2)),0,1)#602
    data3=np.zeros((25,40))#初始化最终值
    for row in range(25):
        for col in range(40):
            if data1[row,col]+data2[row,col]==2:
                data3[row,col]=1
            elif data1[row,col]+data2[row,col]==1:
                ran=np.random.rand()
                if ran>0.5:
                    data3[row,col]=1

    data_saver2(data3,path3)

def union2(path1,path2,path3):#这个函数会把两个气敏器件响应结果并集的部分提取出来，差集合并法
    data1=np.delete(np.array(pd.read_excel(path1)),0,1)#611
    data2=np.delete(np.array(pd.read_excel(path2)),0,1)#602
    data3=np.zeros((25,40))#初始化最终值
    for row in range(25):
        for col in range(40):
            if data1[row,col]-data2[row,col]!=0:
                data3[row,col]=1


    data_saver2(data3,path3)
def main():
    # re_test()
    #以下分段，按段执行
#将spikes输送到RC中
    #-------------------这一段是运行RC
    # for i in range(16):
    #     spikepath="E:/新建文件夹/工作/SSI/gasdata/Project/spikes/spike_%d.xlsx"%i
    #     featurepath="E:/新建文件夹/工作/SSI/gasdata/Project/features/feature_%d.xlsx"%i
    #     output_path="E:/新建文件夹/工作/SSI/gasdata/Project/RC_output/output_%d.xlsx"%i
    #     spikes=np.array(pd.read_excel(spikepath)).T#读取的时候按行读取，保存的时候按列保存，pandas真有你的啊
    #     spikes=np.delete(spikes,0,0)
    #     RC1=rc(device_num= spikes.shape[0])
    #     RC1.new_run_rc(spikes,featurepath)
    #     RC1.Virtual_node(featurepath,output_path)
    #     print('已完成：%.2f%%'%(((i+1)/16)*100))
    # print('全部完成')#feature是横着保存的，也可以直接横着读取
    #--------------------这一段是把想要的两个气敏器件的数据拿出来
    # for i in range(16):
    #     datapath="E:/新建文件夹/工作/SSI/gasdata/Project/RC_output/output_%d.xlsx"%i
    #     savepath1='E:/新建文件夹/工作/SSI/gasdata/Project/trainset/TGS2611_%d.xlsx'%i
    #     savepath2='E:/新建文件夹/工作/SSI/gasdata/Project/trainset/TGS2602_%d.xlsx'%i
    #     trainset(datapath,savepath1,savepath2)
    #     print('已完成：%.2f%%'%(((i+1)/16)*100))
    # print('全部完成')
    #-------------------这一段是展示画图
    # drawpath='E:/新建文件夹/工作/SSI/gasdata/Project/trainset/TGS2611_0.xlsx'
    # drawpath2='E:/新建文件夹/工作/SSI/gasdata/Project/trainset/TGS2602_0.xlsx'
    # arr1=np.delete(np.array(pd.read_excel(drawpath)),0,1)[:,29]
    # arr2=np.delete(np.array(pd.read_excel(drawpath2)),0,1)[:,29]
    # plot_map(arr1)
    # plot_map(arr2)
    #-------------------将两个气敏器件并集的部分作为唯一的输入
    # path1='E:/新建文件夹/工作/SSI/gasdata/Project/trainset/TGS2611_0.xlsx'
    # path2='E:/新建文件夹/工作/SSI/gasdata/Project/trainset/TGS2602_0.xlsx'
    # path3='E:/新建文件夹/工作/SSI/gasdata/Project/trainset-subtract/union_0.xlsx'
    # union2(path1,path2,path3)
    #------------------将并集结果展示出来
    # drawpath = 'E:/新建文件夹/工作/SSI/gasdata/Project/trainset-subtract/union_0.xlsx'
    # for i in range(10):
    #     arr=np.delete(np.array(pd.read_excel(drawpath)),0,1)
    #     plot_map(arr[:,i],'Ethylene',i)
    #     plot_map(arr[:,i+10],'Ethanol',i)
    #     plot_map(arr[:, i + 20], 'Carbon monoxide', i)
    #     plot_map(arr[:, i + 30], 'Methane', i)
#结束

#将hybirdspikes输入RC中
    #-------------------这一段是运行RC
    # for i in range(16):
    #     spikepath="E:/新建文件夹/工作/SSI/gasdata/Project/hybirdspikes/hybirdspike_%d.xlsx"%i
    #     featurepath="E:/新建文件夹/工作/SSI/gasdata/Project/hybirdfeatures/hybird_feature_%d.xlsx"%i
    #     output_path="E:/新建文件夹/工作/SSI/gasdata/Project/hybirdRC_output/hybird_output_%d.xlsx"%i
    #     spikes=np.array(pd.read_excel(spikepath)).T#读取的时候按行读取，保存的时候按列保存，pandas真有你的啊
    #     spikes=np.delete(spikes,0,0)
    #     RC1=rc(device_num= spikes.shape[0])
    #     RC1.new_run_rc(spikes,featurepath)#features是横着保存的，只是一段中间值而已
    #     RC1.Virtual_node(featurepath,output_path)
    #     print('已完成：%.2f%%'%(((i+1)/16)*100))
    # print('全部完成')
    #--------------------这一段是正则化归到0~1之间
    # for i in range(16):
    #     datapath="E:/新建文件夹/工作/SSI/gasdata/Project/hybirdRC_output/hybird_output_%d.xlsx"%i
    #     savepath='E:/新建文件夹/工作/SSI/gasdata/Project/hybirdtrainset/hybirdtrainset_%d.xlsx'%i
    #     hybirdtrainset(datapath,savepath)
    #     print('已完成：%.2f%%'%(((i+1)/16)*100))
    # print('全部完成')
    #------------------将hybird结果展示出来
    drawpath = 'E:/新建文件夹/工作/SSI/gasdata/Project/hybirdtrainset/hybirdtrainset_0.xlsx'
    for i in range(10):
        arr=np.delete(np.array(pd.read_excel(drawpath)),0,1)
        plot_map(arr[:,i],'Ethylene',i)
        plot_map(arr[:,i+10],'Ethanol',i)
        plot_map(arr[:, i + 20], 'Carbon monoxide', i)
        plot_map(arr[:, i + 30], 'Methane', i)


if __name__ == '__main__':
    main()
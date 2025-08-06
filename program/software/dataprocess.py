import os
from statistics import median
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_time = np.arange(0, 300, 0.1)  # 为作图作准备
folderpath = "E:\新建文件夹\工作\\SSI\gasdata\\Project\\dataset1\\sample_data_0.xlsx"


def datashow(folderpath):  # 为了确定数据库里谁是谁，不用多管
    row_data = np.array(pd.read_excel(folderpath))
    plt.figure()
    for n in range(32):
        p_data = []
        for i in range(10):
            p_data.append(row_data[:, i + 1 + 10 * n])
            plt.subplot(8, 4, n + 1)
            plt.plot(x_time, p_data[i], label=f'group {i + 1}')
            plt.ylim(0.4, 1.1)

    plt.show()


def Exponential_smoothing(x):  # 指数平滑法，慎用
    x_num = x.shape[0] - 1
    s = np.zeros(x_num + 1)
    s[0] = x[0]
    for i in range(x_num):
        s[i + 1] = 0.1 * x[i + 1] + 0.9 * s[i]
    return s


def mean_filter(x):  # 均值滤波
    x_num = x.shape[0] - 1
    s = np.zeros(x_num + 1)
    s[0] = x[0]
    for i in range(x_num - 1):
        s[i + 1] = (x[i] + x[i + 1] + x[i + 2]) / 3
    s[x_num] = x[x_num]
    return s


def median_filter(x):  # 中位数滤波
    x_num = x.shape[0] - 1
    s = np.zeros(x_num + 1)
    s[0] = x[0]
    for i in range(x_num - 1):
        s[i + 1] = median([x[i], x[i + 1], x[i + 2]])
    s[x_num] = x[x_num]
    return s


def smooth_data(folderpath=folderpath, col=149):  # 得到滤波数据
    row_data = np.array(pd.read_excel(folderpath))[:, col]
    s_data1 = Exponential_smoothing(row_data)
    S=SNR(row_data,s_data1)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x_time,row_data)
    plt.scatter(50,row_data[499],color='r')
    plt.subplot(2,1,2)
    plt.text(220,0.95,'SNR=%.2f'%S)
    plt.plot(x_time,s_data1)
    plt.scatter(50, s_data1[499], color='r')
    plt.show()
    return s_data1


def SNR(row_data, s_data):  # 看一下信噪比是多少
    n_data = []
    for i in range(len(row_data)):
        n_data.append(abs(row_data[i] - s_data[i]))
    output = 10 * np.log10(sum(s_data) / sum(n_data))  # 这是信噪比的公式，信号的功率和噪声的功率
    return output


def response(data):
    sample_rate = 1  # 时间窗口
    segment = int(3000 / sample_rate)
    n_data = np.zeros(segment)
    for i in range(segment):
        n_data[i] = np.mean(data[i * sample_rate:(i + 1) * sample_rate])
    base = np.mean(data[0:499])  # 50s作为基准线
    res = n_data / base
    return res


def alarm(data):
    sample_rate = 1  # 时间窗口
    segment = int(3000 / sample_rate)
    n_data = np.zeros(segment)
    for i in range(segment):
        n_data[i] = np.mean(data[i * sample_rate:(i + 1) * sample_rate])
    base = np.mean(data[0:499])  # 50s作为基准线
    res = n_data / base
    tri = 0
    ala = [c if c > 1.003 else 0 for c in res]
    for i in range(segment):  # 找到报警信号的坐标
        if ala[i] != 0 and x_time[i] >= 50:
            tri = i
            break
    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.text(75,0.95,'baseline=%.2f'%base)
    # plt.plot(x_time,n_data)
    # plt.subplot(3,1,2)
    # plt.text(75,0.95,'Response')
    # plt.plot(x_time,response)
    # plt.subplot(3,1,3)
    # plt.annotate('alarm at %.2f s'%x_time[tri],(52.6,1),xytext=(100,0.5),
    #              arrowprops=dict(arrowstyle = '->',connectionstyle = 'arc3,rad=0.2'))
    # plt.plot(x_time,ala)
    # plt.show()
    return res, tri


def spikes(response, tri):  # 100个点，前50个动态编码，后50个泊松编码，最后形成一个4s的脉冲序列
    v = np.zeros(400)  # 30s的速度
    d = np.mean(response[tri + 200:tri + 300])  # 20-30s内的响应,也是泊松编码的概率值
    p = np.random.random(50)
    # a=np.zeros(300)
    for n in range(400):
        v[n] = response[tri + n + 1] - response[tri + n]
    # for n in range(299):
    #     a[n]=v[n+1]-v[n]
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(v)
    # plt.subplot(2,1,2)
    # plt.plot(a)
    # plt.show()
    spike = np.zeros(100)
    for i in range(50):  # 动态编码
        if np.mean(v[4 * i:4 * i + 4]) <= -0.001:
            spike[i] = 1
    for i in range(50):  # 泊松编码
        if p[i] >= d:
            spike[i + 50] = 1
    # x = np.linspace(0, 4, 201)  # 4s的脉冲序列，每个间隔20ms,为了作图而已
    # spike_train = np.zeros(201)
    # for i in range(spike.shape[0]):
    #     spike_train[2 * i + 1] = spike[i]
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(v)
    # plt.subplot(2, 1, 2)
    # plt.plot(x, spike_train)
    # plt.show()
    return spike

def hybirdspikes(inputpath,outputpath):#把TGS2611和TGS2602的响应速度提取出来
    data1=np.delete(np.array(pd.read_excel(inputpath)),0,1)
    data2=np.zeros((100,40))
    for i in range(50):
        data2[i]=data1[i][0:40]
        data2[i+50]=data1[i][120:160]
    data_saver(data2,outputpath)

def data_saver(file, path):
    # 将数据保存进excel
    data = pd.DataFrame(file)
    writer = pd.ExcelWriter(path)
    data.to_excel(writer)
    writer.close()

def tag_lable(path,i=0):#给hybird训练集打个标签
    img=np.delete(np.array(pd.read_excel(path)).T,0,0)
    new=np.zeros((40,26))
    for row in range(40):
        for line in range(25):
            new[row][line]=img[row][line]
        new[row][25]=int(row/10)
    data_saver(new.T,'hybirdtrainset/hybird_label_%d.xlsx'%i)



def main():
    # datashow("E:\新建文件夹\工作\\SSI\gasdata\\Project\\dataset1\\sample_data_0.xlsx")
    #---------------------------速度响应和脉冲响应
    # data_spike = np.zeros((320, 100))  # 初始化,因为sample中就有320列，每列数据生成一串脉冲。np.array不能动态加元素,非得先初始化
    # s_data = smooth_data(folderpath, 40)  # 只用最强烈的那个数据作为报警信号，其他不考虑
    # res, tri = alarm(s_data)
    # for t in range(1, 16):#这个循环把所有的sample文件转成spike文件
    #     folderpath = "E:\新建文件夹\工作\\SSI\gasdata\\Project\\dataset1\\sample_data_%d.xlsx" % t
    #     for n in range(320):  # 保存脉冲数据，懒得再去写一个函数了，直接放这算了
    #         s_data2 = smooth_data(folderpath, n + 1)
    #         res2 = response(s_data2)
    #         mid = spikes(res2, tri)
    #         for i in range(100):
    #             data_spike[n][i] = mid[i]
    #         print('processing................................第%d个文件，第%d个数据,小节进展：%.2f%%' % (t,n + 1, ((n + 1) / (320)) * 100))
    #     data_saver(data_spike.T, 'E:/新建文件夹/工作/SSI/gasdata/Project/spikes/spike_%d.xlsx' % t)
    #     print('spike trains saved successfully>>>>>>>>> 第%d个文件，总进展%.2f%% ' %(t,(t/15)*100))
    # print('saved all')
    #-------------------这一段将TGS2611和TGS2602的响应速度重构成一个脉冲
    # for i in range(16):
    #     inputpath='E:/新建文件夹/工作/SSI/gasdata/Project/spikes/spike_%d.xlsx'%i
    #     outputpath='E:/新建文件夹/工作/SSI/gasdata/Project/hybirdspikes/hybirdspike_%d.xlsx'%i
    #     hybirdspikes(inputpath,outputpath)
    #------------------这一段给训练集打标签
    # for i in range(16):
    #     path='hybirdtrainset/hybirdtrainset_%d.xlsx'%i
    #     tag_lable(path,i)
    #----------------画图的一些代码
    data_saver(smooth_data(),"E:/新建文件夹/工作/SSI/gasdata/Project/毕业论文/SNR.xlsx")
    row_data = np.array(pd.read_excel(folderpath))[:, 1]
    data_saver(row_data,"E:/新建文件夹/工作/SSI/gasdata/Project/毕业论文/Row.xlsx")
if __name__ == '__main__':
    main()

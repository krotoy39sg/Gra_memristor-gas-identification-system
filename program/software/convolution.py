import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.nn
import os

def HUST():
    img = np.zeros((4, 25))
    img[0] = [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]  # H
    img[1] = [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # U
    img[2] = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  # S
    img[3] = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]  # T

    f, ax = plt.subplots()
    for i in range(4):

        plt.subplot(1, 4, i+1)
        plt.imshow(img[i].reshape((5, 5)))
        plt.axis('OFF')

    # # y 轴不可见
    # frame.axes.get_yaxis().set_visible(False)
    # # x 轴不可见
    # frame.axes.get_xaxis().set_visible(False)

    ax.spines['top'].set_color('none')  # 隐藏顶部边框
    ax.spines['right'].set_color('none')  # 隐藏右侧边框
    ax.spines['bottom'].set_color('none')  # 隐藏底部边框
    ax.spines['left'].set_color('none')  # 隐藏左侧边框
    plt.show()

    return img

class synapse:#这是卷积的突触，并非SNN中的突触
    def __init__(self, kernel_size, kernel_num, train_or_test=0, G_name=r'', channels=3):
        #下载突触器件数据集
        self.LTP = np.array(pd.read_excel('E:/BaiduNetdiskWorkspace/mem+gas/gasdata/Project/Gra_TaOx_ZnO/LTP.xlsx',
                                          header=None))[:, 0]
        self.LTD = np.array(pd.read_excel('E:/BaiduNetdiskWorkspace/mem+gas/gasdata/Project/Gra_TaOx_ZnO/LTD.xlsx',
                                          header=None))[:, 0]
        #线性仿真
        self.LTP = np.linspace(self.LTP[0], self.LTP[-1], 200)
        self.LTD = np.linspace(self.LTD[-1], self.LTD[0], 200)

        # 输入和输出的维度决定了突触阵列的规模
        # 暂时先用一个卷积核
        self.kernel_size = kernel_size    # 卷积核的size
        self.kernel_num = kernel_num      # 卷积核数目
        self.kernel_step = 1              # 卷积步数为1
        self.kernel_padding = 0           # 卷积填充为0
        self.V_shape = 28                 # 数据类型为28*28
        self.pooling_size = 2             # 池化层的size是2

        self.dt = 1e-6                    # 一个时间步长为 1 us
        self.due = 0.5                    # 占空比为 0.5
        self.V = 0.13                     # 一个脉冲的电压为 0.13V
        # self.G_unit = 1e-6                # 电导的单位是1e-6

        # 定义一个权重的改变量w,数据结构是4*8,卷积核的size是2*2,卷积核的数目是8
        self.w = np.zeros((np.square(self.kernel_size), self.kernel_num))
        # 构建阵列，即每个器件的电导初始化
        self.G = np.zeros((np.square(self.kernel_size), self.kernel_num))

        # 初始化卷积核，暂时将卷积核的权重完全随机化,train_or_test=0时为训练模式,train_or_test为1时为测试模式;G_name是存储好的权重阵列
        if train_or_test == 0:
            self.G = np.max(self.LTP) - np.random.random((np.square(self.kernel_size), self.kernel_num)) * (
                    np.max(self.LTP) - np.min(self.LTP)) * 1
        else:
            self.G = np.genfromtxt(G_name, delimiter=',') # 这是为了导入数据

        # # 将所有的突触权重做成一个等差数列
        # # 将Pd_W_WO3_Pd突触的权值初始化为35个电导值,暂时不使用这种方法，因为器件可能达不到如此的线性度和对称性
        # self.synapse_G = np.linspace(np.min(self.LTP), np.max(self.LTP), 200)
        # self.G_num = np.random.randint(20, high=35, size=self.kernel_size * self.kernel_size * self.kernel_num,
        # dtype='l')
        # # 将self.G_num reshape成一个和突触阵列相同的矩阵，此处为(25,8)
        # self.G_num = np.reshape(self.G_num, (self.kernel_size * self.kernel_size, self.kernel_num))
        # # 设置输入神经元的输出频率阈值,输入神经元的输出频率500000
        # self.f_input_th = 4.5e5

        # for i in range(self.kernel_size * self.kernel_size):
        #     for j in range(self.kernel_num):
        #         self.G[i, j] = self.synapse_G[self.G_num[i, j]] * self.G_unit

        # 构建阵列，即每个器件的电导初始化
        # self.G = np.random.random((self.input_number, self.output_number)) * (np.max(self.LTP) - np.min(self.LTP))
        # 采样电阻设置为0.1S
        self.G_s = 0.1
        self.magnify = 2850 # 超参数采样信号放大200倍,MNIST的(5,5)的卷积核采用450,cifar100采用28，气敏的数据可能得更大

        # 设置输入神经元阈值频率
        self.input_f_th_mean = 1
        self.input_f_th = self.input_f_th_mean * np.ones(self.kernel_num)

    def VMM_operation(self, V):
        # 采用忆阻阵列实现VMM运算
        # 输入为电压信号；输出为放大后的采样电阻电压信号

        spike_line, spike_queue = V.shape
        # spike_line是数据个数，spike_queue是数据长度

        G_line, G_queue = self.G.shape
        output = np.zeros((spike_line, G_queue))

        for line in range(spike_line):
            for queue in range(G_queue):
                G_sum = sum(self.G[:, queue])
                output[line, queue] = sum(V[line, :] * self.G[:, queue]) / (G_sum + self.G_s) * self.magnify

        return output

    # def VMM(self, V, G_one):
    #     # 采用忆阻阵列实现VMM运算
    #     # 只需要做加和,并不需要乘以放大倍数
    #     # 输入为电压信号；输出为放大后的采样电阻电压信号
    #
    #     spike_line, spike_queue = V.shape
    #     G_line, G_queue = G_one.shape
    #     output = np.zeros((spike_line, G_queue))
    #
    #     for line in range(spike_line):
    #         for queue in range(G_queue):
    #             G_sum = sum(G_one[:, queue])
    #             output[line, queue] = sum(V[line, :] * G_one[:, queue])
    #
    #     return output

    def cnn(self, V):
        # 先确定将signal的感受野

        # 卷积核滑动的步长为1,填充为0,5*5的数据，卷积核2*2,卷积后的矩阵4*4
        # 将output做一个空array暂存
        image_size_1D, time = V.shape   # 每个V应该是个被拉平的数据，多个被拉平的数据组合在一起又会是一个新阵列
        image_size = int(math.sqrt(image_size_1D))

        # 可以但是没必要而且会降低代码的复用性
        V = np.reshape(V, (image_size, image_size))# 把拉平的数据重构成新矩阵

        # 首先计算输出图的维度：输入维度+填充-卷积核维度/卷积步长+1
        output_size = int((image_size + self.kernel_padding - self.kernel_size / self.kernel_step) + 1)

        # 使用八个卷积核,需要使用一个三维的矩阵
        output = np.zeros((self.kernel_num, output_size, output_size))

        # 需要一个记录感受野的变量，行数是感受野数据拉平后的长度，列数是输出长度
        filed_time = np.zeros((np.square(self.kernel_size), np.square(output_size)))

        # 八个卷积核用八个srm神经元,输入时长为(1000*24*24，8)
        # 换一种形式这个代码实现太过于复杂
        # output = np.zeros((output_size * output_size * time, self.kernel_num))
        # cache = np.zeros((time, self.kernel_size))
        for i in range(output_size):
            for j in range(output_size):
                # 感受野大小与卷积核大小相同，卷积核前进一个步长，感受野更新，此处的time为1
                filed = np.zeros((self.kernel_size, self.kernel_size))
                for ker_len in range(self.kernel_size):
                    for ker_queue in range(self.kernel_size):
                        # 这一行V的选取有两个方面：1是i，j也就是卷积核滑动的选取，2是kernel_queue和ker_len的选取
                        filed[ker_len, ker_queue] = V[ker_len+i, ker_queue+j]

                filed_time[:, j + i * output_size] = filed.reshape((-1, 1)).T

                # cache = self.VMM_operation(filed.reshape((-1, 1)))
                # time_cache_0 = (output_size * i + j) * time
                # time_cache_1 = (output_size * i + j + 1) * time
                output[:, i, j] = self.VMM_operation(filed.reshape((-1, 1)).T)
        # 使用忆阻器实现卷积

        # 卷积核的kernel_size即为感受野
        # plt.figure()
        # plt.subplot(211)
        # plt.imshow(V)
        # plt.axis('off')                                                     # 将绘图的坐标轴隐藏
        # plt.subplot(212)
        # plt.imshow(output[0, :, :])
        # plt.axis('off')
        # plt.show()

        return output, filed_time               # output是卷积之后的结果，filed_time是感受野里的输入数据

    def cnn_CIFAR100(self, V):#目前用不上
        # cnn_CIFAR100一次处理一个批次的所有图像
        images_num, channels, image_x, image_y = V.shape

        output_x = int((image_x + self.kernel_padding - self.kernel_size / self.kernel_step) + 1)
        output_y = int((image_y + self.kernel_padding - self.kernel_size / self.kernel_step) + 1)
        # self.kernel_size指一个通道的卷积核,channels指通道数,output_x和output_y分别指横坐标和纵坐标
        output = np.zeros((images_num, channels, self.kernel_num, output_x, output_y))
        # self.kernel*self.kernel为reshape(-1,1)后的filed,output_x*output_y为时间步长
        filed_records = np.zeros((images_num, channels, self.kernel_size*self.kernel_size, output_x * output_y))

        for i in range(images_num):
            for j in range(channels):
                # 需要使用归一化函数将输入归一化
                output[i, j, :, :, :], filed_records[i, j, :, :] = self.cnn(V[i, j, :, :].reshape([-1, 1]))

                # for k in range(output_x):
                #     for m in range(output_y):
                #         filed = torch.zeros((self.kernel_size, self.kernel_size))
                #
                #         for ker_x in range(self.kernel_size):
                #             for ker_y in range(self.kernel_size):
                #                 filed[ker_x, ker_y] = V[i, j, ker_x+k, ker_y+m]
                #
                #         filed_records[i, j, :, :, m+(k*output_y)] = filed
                #         output[i, j, :, k, m] = self.VMM_operation(filed.reshape((-1, 1)).T)
        return output, filed_records

    def dynamic_thresholds(self, filed):#调整阈值
        kernel_size, t = filed.shape
        filed[filed >= 1] = 1
        self.dynamic_t = np.sum(filed, axis=0)

    def dynamic_SRDP(self, filed, feature_map):
        kernel_size, t = filed.shape
        kernel_num, t = feature_map.shape
        for i in range(t):
            for j in range(kernel_num):
                if feature_map[j, i] >= self.dynamic_t[i]:
                    for k in range(kernel_size):
                        if filed[k, i] >= 1:
                            self.w[k, j] += 4
                        else:
                            self.w[k, j] -= 2
                else:
                    for k in range(kernel_size):
                        if filed[k, i] >= 1:
                            self.w[k, j] -= 1
                        else:
                            self.w[k, j] += 1

    def pooling(self,V):
        # 池化层可以用tf来实现
        # 池化层，池化需要将每一个SRM神经元的V输入一次
        V_num, V_line, V_queue = V.shape                    # 卷积核数量,行,列
        filed = np.zeros((V_num, self.pooling_size, self.pooling_size))
        output_size_line = V_line//self.pooling_size                         # 注意经过池化层后的ouput需要能被pooling_size整除
        output_size_queue = V_queue//self.pooling_size                       # 注意经过池化层后的ouput需要能被pooling_size整除
        output = np.zeros((V_num, output_size_line, output_size_queue))
        for num in range(V_num):                                             # 卷积核数量
            for i in range(output_size_line):
                for j in range(output_size_queue):
                    filed = np.zeros((V_num, self.pooling_size, self.pooling_size))
                    # 用一个稍微笨一点的遍历方法便于代码的编写
                    for k in range(self.pooling_size):
                        for m in range(self.pooling_size):
                            filed[num, k, m] = V[num, (i * self.pooling_size) + k, (j * self.pooling_size) + m]
                    # 最大值池化
                    output[num, i, j] = np.max(filed)

        # plt.figure()
        # plt.imshow(output[0, :, :])
        # plt.axis('off')
        # plt.show()
        return output

    def refractory_image(self, V):
        # 实际上是Deconvolution,也就是反卷积,将图片重构之后后检查是否提取到足够的特征
        pass

    def SRDP(self, filed, feature_map):
        # num为获取的第几个输出神经元的输出值
        # 先获取神经元个数
        # output_num = output_spikes.shape[1]
        # # 获取输入神经元的输入时间步长
        # input_num, time_legth = input_spikes.shape
        # 计算输入神经元的输出频率
        # f_input = np.sum(input_spikes, axis=1) / (time_legth * self.dt * self.V)
        # 编写SRDP法则
        # input_number, time_i = input_spikes.shape[0]
        # output_number, time_o = output_spikes.shape[0]
        # 定义一个w为改变权重

        # image_size_1D, time = input_spikes.shape
        # image_size = int(math.sqrt(image_size_1D))

        # 可以但是没必要而且会降低代码的复用性
        # V = np.reshape(input_spikes, (image_size, image_size))

        # 首先计算输出图的维度：输入维度+填充-卷积核维度/卷积步长+1
        # output_size = int((image_size + self.kernel_padding - self.kernel_size / self.kernel_step) + 1)
        # input_s = np.zeros((self.kernel_num, output_size, output_size))

        # 使用八个卷积核,需要使用一个三维的矩阵
        # SRDP需要定义一个权值为1的卷积核,卷积核维度应该是(25,8)来完成卷积操作,做出需要的input_spikes
        # one_kernel = np.ones((self.kernel_size * self.kernel_size, self.kernel_num))

        # for i in range(output_size):
        #     for j in range(output_size):
        #         # 感受野大小与卷积核大小相同，卷积核前进一个步长，感受野更新，此处的time为1
        #         filed = np.zeros((self.kernel_size, self.kernel_size))
        #         for ker_len in range(self.kernel_size):
        #             for ker_queue in range(self.kernel_size):
        #                 # 这一行V的选取有两个方面：1是i，j也就是卷积核滑动的选取，2是kernel_queue和ker_len的选取
        #                 filed[ker_len, ker_queue] = V[ker_len+i, ker_queue+j]
        #         input_s[:, i, j] = self.VMM(filed.reshape((-1, 1)).T, one_kernel)
        #
        # input_s = self.pooling(input_s)         # 池化
        filed_size, time = filed.shape            # filed_size应该为25,time为576;feature_map的维度为(8,576)
        for i in range(time):
            for j in range(self.kernel_num):
                if feature_map[j, i] >= 1:
                    for k in range(filed_size):
                        if filed[k, i] >= self.input_f_th_mean:
                            self.w[k, j] += 4
                        else:
                            self.w[k, j] -= 2
                else:
                    for k in range(filed_size):
                        if filed[k, i] >= self.input_f_th_mean:
                            self.w[k, j] -= 1
                        else:
                            self.w[k, j] += 1

        # plt.figure()
            # if output_spikes[i] >= 1:
            #     for j in range(self.kernel_num):
            #         if input_spikes[j] > self.input_f_th_mean:
            #             self.w[j, i] += 1
            #         else:
            #             self.w[j, i] -= 2
            # else:
            #     for j in range(np.square(self.kernel_size)):
            #         if input_spikes[j] > self.input_f_th_mean:
            #             self.w[j, i] -= 1
            #         else:
            #             # 这是师兄独有的一行代码,除了他的SRM神经元别的都不能用巨坑
            #             self.w[j, i] += 0

    def SRDP_CIFAR100(self, filed, featuremap):#师兄对CIFAR100的识别，这里看看就行
        batchs_images, channels, t, SRM_neurons = featuremap.shape
        batchs_images, channels, kernel_size_square, t = filed.shape
        for i in range(batchs_images):
            for j in range(channels):
                # 真的纯sb,写代码一定要规范不然就会遇到这种sb事情
                self.SRDP(filed[i, j, :, :], featuremap[i, j, :, :].T)

    def Update(self):
        # 更新电导值的函数在cnn中是适用的
        # 这个方法更新电导值
        for i in range(np.square(self.kernel_size)):
            for j in range(self.kernel_num):
                if self.w[i, j] > 0:
                    err = np.abs(self.G[i, j] - self.LTP)
                    min_err = np.where(np.min(err) == err)[0]
                    p = int(min_err + self.w[i, j])
                    if abs(p) >= self.LTP.shape[0]:
                        self.G[i, j] = self.LTP[-1]
                    else:
                        self.G[i, j] = self.LTP[p]
                else:
                    err = np.abs(self.G[i, j] - self.LTD)
                    min_err = np.where(np.min(err) == err)[0]
                    p = int(min_err + self.w[i, j])
                    if abs(p) >= self.LTD.shape[0]:
                        self.G[i, j] = self.LTD[-1]
                    else:
                        self.G[i, j] = self.LTD[p]

class input_neurons:
    def __init__(self):
        self.pixel_num = 28 * 28                         # 输入神经元的个数
        self.dt = 1e-6                                   # 时间步长1 us
        self.pulse_t = 2e-4                              # 一个脉冲的持续时间1 ms
        self.pulse_num = 5                               # 一次激发的脉冲数目
        self.V = 0.13                                    # 脉冲的电压值
        self.due = 0.5                                   # 脉冲的占空比为0.5
        self.frequency = 10                              # 脉冲频率为10
        self.T = int(self.pulse_t / self.dt)             # 脉冲的周期为100
        # 计算一张图输入编码所需要的时间
        self.encode_time = int(self.pulse_num * self.T)  # 所有的脉冲所需要的编码时间

        # 制作一个空集合存储编码
        # 计算时间例如：batch: 20, spike长度为25*20
        self.spike = np.zeros((self.pixel_num, self.encode_time))                # 此处的输出脉冲应该为25 * 500

    def binary(self,image):
        # 二值化图像，灰度值大于200的置为1，灰度值小于200的置为0
        for counter in range(self.pixel_num):
            if image[counter] >= 200:
                image[counter] = 1
            else:
                image[counter] = 0

        return image

    def encoding(self,image):
        #二值化在纯黑白的图里并不需要
        image = self.binary(image)

        #将图像编码
        for p_n in range(self.pixel_num):
            if image[p_n] == 0:
                for i in range(self.encode_time):
                    self.spike[p_n, i] = 0
            elif image[p_n] == 1:
                for i in range(self.pulse_num):
                    for j in range(int(self.T * self.due)):
                        self.spike[p_n, i * self.T + j + 1] = self.V
        return self.spike

class Neurons:
    def __init__(self, neuron_number, batch_num, dt=1e-6):
        self.running_time = 1
        self.dt = dt                                        # 运行时间1us
        self.neuron_number= neuron_number                   # 设定神经元数目

        # 电路参数
        self.R1 = 1e4                                               # R1电阻为10kOhm
        self.R2 = 1e3                                               # R2阻值为100Ohm,R2是采样电阻,尽可能保证采样电阻和忆阻器的低组态在同一数量级
        self.c = 1e-9                                               # 电容为1nF

        # 忆阻器性能
        self.R_h = 1e6                                              # 忆阻器的高阻态（HRS）
        self.R_l = 1e3                                              # 忆阻器的低阻态（LRS）
        self.mean_th = 4.5                                          # 忆阻器阈值的均值
        self.mean_hold = 3.3                                        # 忆阻器保持电压的均值
        self.T_delay = 1e-6                                         # 忆阻器响应时间为1ms
        self.T_hold = 1e-6                                          # 忆阻器弛豫时间为1ms

        # 神经元参数
        self.V_th = self.mean_th * np.ones(self.neuron_number)      # 忆阻器阈值转变电压为4.5V；阈值电压的分布在运行中体现
        self.V_hold = self.mean_hold * np.ones(self.neuron_number)  # 忆阻器保持电压为0.05V
        self.R_m = self.R_h * np.ones(self.neuron_number)           # 忆阻器的阻值调到高阻态
        self.R_timer = -1 * np.ones(self.neuron_number)             # 记录忆阻器响应时间及弛豫时间
        self.th = 1 * np.ones(self.neuron_number)                   # 神经元阈值：设置为0.001V
        self.f_output = np.zeros(self.neuron_number)                # 神经元输出频率初始化为0
        self.f_input = np.zeros(self.neuron_number)                 # 输入频率初始化为0
        self.V_m = np.zeros(self.neuron_number)                     # 神经元膜电位初始化为0
        self.V_s = np.zeros(self.neuron_number)                     # 采样电阻分压
        self.firing = np.zeros(self.neuron_number)                  # 记录点火信息
        self.Record_output = np.zeros((batch_num * 4, self.neuron_number))  # 神经元激发的情况
        self.ouput_timer = 0                                        # 定义一个计数器，用于计数每一个图片输入后神经元的激发情况
        # self.Spikes_num = np.zeros(self.neuron_number)            # 用于记录单个图片中神经元激发个数

        # 不应期，在输出神经元释放脉冲后，不应期会使神经元在下几张图片不再激发
        self.refractory_period_time = 20
        # self.refractory_period = np.zeros(self.neuron_number)
        # 侧向抑制，在输出神经元释放脉冲后，侧向抑制会使其他的输出神经元阈值升高
        self.lateral_inhibition_time = 6
        # self.lateral_inhibition = np.zeros(self.neuron_number)

        # 定义一个时间计数器确定神经元阈值
        self.time_counter = np.zeros(self.neuron_number)
        # 规定离散的神经元阈值
        self.lateral_inhibition_th = [1.2, 0.25, 0.1, 0.005]
        self.refractory_period_th = [3.5, 2, 1, 0.5]

        # 规定输入脉冲的个数以及输入电压
        self.spikes_num = 10
        self.spikes_V = 3
        self.input_spikes_num = np.zeros(self.neuron_number)  # 记录输入脉冲的数量

        # 设计一个state记录神经元的触发情况
        self.state = np.zeros(self.neuron_number)

    def run_neurons(self, input_spikes):
        # 根据输入信号确定时间步长,time_step即为输入脉冲的个数
        # 每一次输入都会将记录置为 0
        # 按时间步长开始运行
        time_step = self.spikes_num
        self.f_output = np.zeros(self.neuron_number)  # 神经元输出频率初始化为0
        Record_V_m = np.zeros((time_step, self.neuron_number))
        Record_V_s = np.zeros((time_step, self.neuron_number))
        Record_V_o = np.zeros((time_step, self.neuron_number))
        Record_V_th = np.zeros((time_step, self.neuron_number))
        Record_R_m = np.zeros((time_step, self.neuron_number))

        # 每一回输入前将忆阻器置于高电阻态
        self.R_m = self.R_h * np.ones(self.neuron_number)  # 忆阻器初始化为高电阻态
        self.V_m = np.zeros(self.neuron_number)  # 神经元膜电位初始化为 0
        self.V_s = np.zeros(self.neuron_number)  # 采样电压初始化为 0

        # 记录所有的神经元释放脉冲的情况
        self.Record_output_spikes = np.zeros((time_step, self.neuron_number))

        # time_step决定了input_spikes将要输入几次
        for i in range(time_step):
            # 根据电路方程计算各参量
            I1 = (input_spikes - self.V_m) / self.R1
            I2 = self.V_m / (self.R_m + self.R2)
            du = (I1 - I2) * self.dt / self.c
            self.V_m += du
            self.V_s = self.V_m / (self.R_m + self.R2) * self.R2

            # 根据采样电阻分压确定神经元是否激发
            self.firing[self.V_s > self.th] = 1

            # 判断忆阻器是否发生阈值转变
            # 判断是否有多个神经元同时激发，如果有就选择电压信号最大的作为此次输出
            if np.sum(self.firing) > 1:
                p = np.where(self.V_s == np.max(self.V_s))[0][0]    # 后面两个零是为了取到值
                self.firing = np.zeros(self.neuron_number)
                self.firing[p] = 1
                self.f_output[p] += 1  # 记录神经元的输出频率

                # 如果记录状态state所对应的神经元为0
                if self.state[self.firing == 1] == 0:
                    # 激发的神经元进入不应期
                    self.time_counter[self.firing == 1] = self.refractory_period_time

                    # 激发的神经元进入不应期后将其状态量调整为1
                    self.state[self.firing == 1] = 1
                    # 未激发的神经元进入侧向抑制
                    self.time_counter[self.state == 0] = self.lateral_inhibition_time

            # 使用SRM神经元点火后，在两个(固定)时间步长内应该输入不应期及侧向抑制，固定时间步长后应该恢复静息电位输入
            # 阈值更新
            # 如果不需要使用阈值更新属性，可注释掉以下四行
            # if np.sum(self.firing) == 1:
            #     self.th[self.firing != 1] = 0.001 * 20
            #     self.th[self.firing == 1] = 0.001 * 40
            # self.threshold_adjust()

            # 判断忆阻器是否发生阈值转变
            # 此处的V_m正确应该为V2
            V_memristor = self.V_m - self.V_s
            for j in range(self.neuron_number):
                # 阈值：1.电压超过阈值；2.忆阻器处于HRS；3.电压超过阈值的时间大于响应时间
                if V_memristor[j] > self.V_th[j] and self.R_m[j] == self.R_h:
                    if self.R_timer[j] == -1:
                        self.R_timer[j] = int(self.T_delay / self.dt)
                    elif self.R_timer[j] > 0:
                        self.R_timer[j] -= 1
                    elif self.R_timer[j] == 0:
                        self.R_m[j] = self.R_l
                        self.R_timer[j] = -1

                        # 此处是忆阻器自身的阈值不稳定
                        self.V_th[j] = np.random.normal(self.mean_th, 1e-3, 1)

                # 保持：1.电压低于保持电压；2.忆阻器处于LRS；3.电压低于保持电压的时间大于弛豫时间
                if V_memristor[j] < self.V_hold[j] and self.R_m[j] == self.R_l:
                    if self.R_timer[j] == -1:
                        self.R_timer[j] = int(self.T_hold / self.dt)
                    elif self.R_timer[j] > 0:
                        self.R_timer[j] -= 1
                    elif self.R_timer[j] == 0:
                        self.R_m[j] = self.R_h
                        self.R_timer[j] = -1
                        self.V_hold[j] = np.random.normal(self.mean_hold, 1e-3, 1)

            # 记录神经元各节点电压
            Record_V_m[i] = self.V_m
            Record_V_s[i] = self.V_s
            Record_V_o[i] = self.firing
            Record_R_m[i] = self.R_m
            Record_V_th[i] = self.th

            # self.firing = np.zeros(self.neuron_number)
            # 如果神经元有激发，则启用SRM功能,将神经元的激发记录下来
            # if np.sum(self.firing) >= 1:
            #     self.SRM(Record_V_o)

        # 统计四个神经元一共的激发脉冲数目
        # Spikes_num = np.sum(Record_V_o, axis=0)
        # 不应期的设置，使处在不应期的的神经元不能激发
        # 如果神经元激发，则选取与阈值电压差距最大的神经元记为输出神经元
        # if np.sum(self.firing) >= 1:
        #     det = self.V_s - self.th
        #     self.firing[det < np.max(det)] = 0
        #
        #     self.f_output[self.firing == 1] += 1  # 记录输出神经元的输出频率
        #
        #
        for j in range(self.neuron_number):
            # 将不应期和侧向抑制分开
            # self.firing只能记录一次激发的情况而self.state能记录在一个batch中输出神经元的激发情况
            if self.time_counter[j] > int(len(self.refractory_period_th)):
                if self.state[j] == 1:
                    self.th[j] = self.refractory_period_th[0]
                else:
                    self.th[j] = self.lateral_inhibition_th[0]
            elif 0 < self.time_counter[j] <= int(len(self.refractory_period_th)):
                if self.state[j] == 1:
                    self.th[j] = self.refractory_period_th[int(len(self.refractory_period_th) - self.time_counter[j])]
                else:
                    self.th[j] = self.lateral_inhibition_th[int(len(self.refractory_period_th) - self.time_counter[j])]
            else:
                self.th[j] = 0.05

        # 将计时器小于零的置为0
        self.time_counter -= 1
        self.time_counter[self.time_counter < 0] = 0

        self.firing = np.zeros(self.neuron_number)

    # 可视化各节点电压的变化过程
    # input_spikes为输入电压，V_m为忆阻器分电压，
    # self.visualize(input_spikes, Record_V_m, Record_V_s, Record_V_th, Record_V_o, Record_R_m)
    # self.visualize(Record_V_m, Record_V_o)

    def threshold_adjust(self):
        # 阈值调节，如果阈值比初始值大，按照一定规律调节阈值
        for i in range(self.neuron_number):
            if self.th[i] > 0.001:
                self.th[i] *= 0.99

    def visualize(self, *Var):

        # 可视化输入变量Var
        time_step = Var[0].shape[0]
        x = np.linspace(0, time_step * self.dt, time_step)
        var_num = len(Var)
        for i in range(var_num):
            plt.figure(i)
            for j in range(self.neuron_number):
                plt.subplot(self.neuron_number, 1, j + 1)
                plt.plot(x, Var[i][:, j])

        # plt.show()
        # 一个神经元一个画布
        # for i in range(self.neuron_number):
        #     plt.figure(i)
        #     for j in range(0, var_num):
        #         plt.subplot(var_num, 1, j + 1)
        #         plt.plot(x, Var[j][:, i])
        #         plt.legend(['N1', 'N2', 'N3'])
        # # 第一个图是每一个神经元的膜电位
        # for i in range(self.neuron_number):
        #     plt.figure(i)
        #     plt.plot(x, Var[0][:, i])
        # # 第二个图是每一个神经元的输出
        # for j in range(self.neuron_number):
        #     plt.figure(j)
        #     plt.plot(x, Var[1][:, j])

        # plt.show()

    def encoding(self, image):          # 时序脉冲编码
        # 编码的意义在于将图像的像素信号转化为电压脉冲输入
        # 使用reshape函数将image转化为(n，1)
        picture = image.reshape((-1, 1))
        p_x, p_y = picture.shape
        spikes = np.zeros((p_x, p_y))
        # 编码规则为如果二值化后的图像像素点为“1”则输出五个脉冲，脉冲为0.1V,如果像素点为0，则输出为0V
        for counter in range(p_x):
            if picture[counter] >= 200:
                spikes[counter, 0] = self.spikes_V
            else:
                spikes[counter, 0] = 0

        return spikes

    def f_encoding(self, image):
        # f_encoding的意义在于改动脉冲输入的频率
        # 使用reshape函数将image转化为(n，1)
        picture = image.reshape((-1, 1))
        p_x, p_y = picture.shape
        # 设计一个空列表用来储存
        spikes = np.zeros((p_x, p_y))

        # 编码规则为如果二值化后的图像灰度大于200则输出五个脉冲，脉冲为0.1V,如果像素点为0，则输出为0V
        for counter in range(p_x):
            if picture[counter] >= 200:
                spikes[counter, 0] = self.spikes_num
            else:
                spikes[counter, 0] = 0

        return spikes

    def encoding_CIFAR100(self, x):
        # 脉高的编码方法,该数据在提取过程中已经标准化,需要归一化
        images_num, channel, pixel_x, pixel_y = x.shape
        x += abs(torch.min(x))
        x /= torch.max(x)
        input_v = x * self.spikes_V
        input_f = x * self.spikes_num

        return input_v, input_f

    def Compilation(self, in_spike):
        # 数据编译
        SRM_num, input_num_line, input_num_queue = in_spike.shape
        output_spikes = np.zeros((SRM_num, input_num_line*input_num_queue))
        for i in range(input_num_line):
            for j in range(input_num_queue):
                output_spikes[:, j + i * input_num_queue] = in_spike[:, i, j]

        return output_spikes.T

    def Compilation_CIFAR100(self, in_put):
        images_num, channels, kernel_num, image_x, image_y = in_put.shape
        # images_num是一个批次中的图像数量,channels是通道数,kernel_num是所需要的神经元数量,image_x*image_y是时间步长
        output_spikes = np.zeros((images_num, channels, image_x*image_y, kernel_num))

        for i in range(images_num):
            for j in range(channels):
                cache = in_put[i, j, :, :, :]
                for k in range(image_x):
                    for m in range(image_y):
                        output_spikes[i, j, m + k * image_y, :] = in_put[i, j, :, k, m]

        return output_spikes

    # def SRM(self, Record_V_o):
        # 关于SRM神经元的一些重要的参数
        # 侧向抑制，暂时不用
        # self.Lateral_inhibition = 5

        # 不应期，暂时不用
        # self.Refractory_period = 5

        # 读出Record_V_o的时间步长
        # time_step = Record_V_o.shape[0]

        # 神经元激发的情况,遍历Record_V_o,发现神经元被激发后，记录并退出循环，即记录第一个激发的神经元
        # 换一种SRM的神经元比较算法，即记录输出脉冲最多的数量
        # for i in range(time_step):
        #     if np.sum(Record_V_o[i, :]) == 1:
        #         self.Record_output[self.ouput_timer] = Record_V_o[i, :]
        #         break
        # self.ouput_timer += 1


def get_image():#cifar
    # 读取标签数据集
    with open(r'C:\Users\czy\Desktop\csnn\MNIST\train-labels-idx1-ubyte\train-labels.idx1-ubyte', 'rb') as lbpath:
        labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    # 读取图片数据集
    with open(r'C:\Users\czy\Desktop\csnn\MNIST\train-images-idx3-ubyte\train-images.idx3-ubyte', 'rb') as imgpath:
        images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols)

    return labels, images


def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    labels, images = get_image()
    cnn_kernel = synapse(5, 8)                                  # 第一个参数是卷积核维度，第二个参数是卷积核个数
    N = Neurons(8, 20)                                          # 初始化神经元,卷积核个数就是连接的SRM神经元个数

    for epoch in range(10):
        plt.figure(epoch)
        for chose_num in range(10):
            # 测试程序，先随机选择一个
            # image = images[chose_num]
            input_v = N.encoding(images[chose_num])

            plt.figure()
            plt.imshow(input_v.reshape((28, 28)))
            plt.show()

            input_f = N.f_encoding(images[chose_num])           # 编码

            # plt.figure()
            # plt.imshow(np.reshape(images[chose_num], (28, 28)))
            # plt.show()

            out_spike, filed = cnn_kernel.cnn(input_v)          # 需要额外使用一个filed
            # out_spike = cnn_kernel.pooling(out_spike)         # 经历卷积层和池化层后输入神经元
            cnn_kernel.dynamic_thresholds(filed)

            out_spike = N.Compilation(out_spike)                # 将数据维度变为8*144
            time, n = out_spike.shape                           # time输入神经元时间维度,n指输入神经元数量

            # 需要一个全局变量,记录神经元的输出信息
            feature_map = np.zeros((N.neuron_number, time))
            for i in range(time):
                N.run_neurons(out_spike[i, :])
                feature_map[:, i] = N.f_output                  # feature_map的维度应该是

            # 如果使用SRDP的话需要将input也做卷积和池化处理
            cnn_kernel.dynamic_SRDP(filed, feature_map)         # 关于冒号的使用是不取最后一个数值的
            # 换种方式plt.show,应该看一个卷积核的
            for j in range(cnn_kernel.kernel_num):
                plt.subplot(2, 4, j + 1)
                plt.imshow(feature_map[j, :].reshape(24, 24))
                plt.axis('off')
            # plt.close()
        # 更新权重
        cnn_kernel.Update()
        plt.show()

    # plt.figure()
    # plt.imshow(np.reshape(out_spike[:, 0], (12, 12)))
    # plt.show()
            # for j in range(N.neuron_number):
            #     plt.subplot(2, 4, j + 1)
            #     plt.imshow(feature_map[j, :].reshape(24, 24))
            #     plt.axis('off')
            # plt.show()

    # plt.figure()
    # plt.imshow(feature_map)
    # plt.show()
    # Spike = input_neurons()      # 初始化输入神经元
    # input_v = Spike.encoding(images[chose_num])
    # out_spike = cnn_kernel.cnn(input_v)


if __name__ == "__main__":
    main()
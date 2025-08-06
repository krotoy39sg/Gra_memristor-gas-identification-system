import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import pandas as pd


class synapse:  # 实际上是突触阵列
    def __init__(self, input_number, output_number):
        self.LTP = np.array(pd.read_excel('E:/新建文件夹/工作/SSI/gasdata/Project/Gra_TaOx_ZnO/LTP.xlsx',
                                          header=None))[:, 0]
        self.LTD = np.array(pd.read_excel('E:/新建文件夹/工作/SSI/gasdata/Project/Gra_TaOx_ZnO/LTD.xlsx',
                                          header=None))[:, 0]
        # 阵列规模
        self.input_number = input_number
        self.output_number = output_number

        self.dt = 1e-6  # 一个时间步长为 1 um
        self.due = 0.5  # 占空比为 0.5
        self.V = 1.5  # 一个脉冲的电压为 0.1V
        # 构建阵列，即每个器件的电导初始化,使用np.random.random的原因是生成(input_number, output_num)的0——1随机数
        self.G = np.max(self.LTP) - np.random.random((input_number, output_number)) * (
                np.max(self.LTP) - np.min(self.LTP)) * 0.1
        # 采样电阻设置为0.1S
        self.G_s = 0.1
        self.magnify = 200  # 采样信号放大200倍
        # 设置输入神经元阈值频率
        self.input_f_th_mean = 3
        self.input_f_th = self.input_f_th_mean * np.ones(input_number)
        # 设置输出神经元阈值频率
        self.output_f_th_mean = 1
        self.output_f_th = self.output_f_th_mean * np.ones(output_number)
        # 定义一个权重的位移量w
        self.w = np.zeros((self.input_number, self.output_number))
        # 将所有的突触权重做成一个等差数列
        # self.synapse_G = np.linspace(np.min(self.LTP), np.max(self.LTP), 35)      # 将突触的权值初始化为35个电导值

        # 此处代码为debug时用，正式用删
        # synapse_G = self.synapse_G

        # self.G_num = np.random.randint(1, high=35, size=self.input_number * self.output_number, dtype='l')

        # 将self.G_num reshape成一个和突触阵列相同的矩阵，此处为(25,4)
        # self.G_num = np.reshape(self.G_num, (self.input_number, self.output_number))

        # 设置输入神经元的输出频率阈值,输入神经元的输出频率500000
        # self.f_input_th = 4.5e5

        # for i in range(self.input_number):
        #     for j in range(self.output_number):
        #         self.G[i, j] = self.synapse_G[self.G_num[i, j]] * self.G_unit

        # 构建阵列，即每个器件的电导初始化
        # self.G = np.random.random((self.input_number, self.output_number)) * (np.max(self.LTP) - np.min(self.LTP))

    def VMM_operation(self, V):
        # 采用忆阻阵列实现VMM运算
        # 输入为电压信号；输出为放大后的采样电阻电压信号

        spike_line, spike_queue = V.shape
        G_line, G_queue = self.G.shape
        output = np.zeros((spike_queue, G_queue))
        # 输出行数为spike的列，输出列数为array的列

        # for line in range(spike_line):
        #     for queue in range(G_queue):
        #         G_sum = sum(self.G[:, queue])
        #         output[line, queue] = sum(V[line, :] * self.G[:, queue])/(G_sum + self.G_s) * self.magnify
        # 注意此处的V只能是一个1列的数据
        for queue in range(G_queue):
            G_sum = sum(self.G[:, queue])
            output[0, queue] = sum(V[:, 0] * self.G[:, queue]) / (G_sum + self.G_s)

        return output * self.magnify

    def SRDP(self, input_spikes, output_spikes):
        # num为获取的第几个输出神经元的输出值
        # 先获取神经元个数
        # output_spikes传过来是一个tuple，纯纯尼玛恶心人
        # output_num = output_spikes.shape[1]
        # # 获取输入神经元的输入时间步长
        # input_num, time_legth = input_spikes.shape
        # 计算输入神经元的输出频率
        # f_input = np.sum(input_spikes, axis=1) / (time_legth * self.dt * self.V)
        # 编写SRDP法则

        # 定义一个w为改变权重
        for i in range(self.output_number):
            if output_spikes[i] >= 1:
                for j in range(self.input_number):
                    if input_spikes[j] > self.input_f_th_mean:
                        self.w[j, i] += 4
                    else:
                        self.w[j, i] -= 2
            else:
                for j in range(self.input_number):
                    if input_spikes[j] > self.input_f_th_mean:
                        self.w[j, i] -= 1
                    else:
                        # 这是师兄独有的一行代码,除了他的SRM神经元别的都不能用巨坑
                        self.w[j, i] += 0

    def STDP(self, input_spikes, output_spikes):
        pass

    #     # 若使用STDP则需要记录第几个脉冲发放
    #     for i in range():

    def Update(self):
        # 这个方法更新电导值
        for i in range(self.input_number):
            for j in range(self.output_number):
                if self.w[i, j] > 0:
                    err = np.abs(self.G[i, j] - self.LTP)
                    min_err = np.where(np.min(err) == err)[0]  # 找出与权重值最相近的位置
                    p = int(min_err + self.w[i, j])
                    if p >= self.LTP.shape[0]:
                        self.G[i, j] = self.LTP[-1]
                    else:
                        self.G[i, j] = self.LTP[p]
                else:
                    err = np.abs(self.G[i, j] - self.LTD)
                    min_err = np.where(np.min(err) == err)[0]
                    p = int(min_err + self.w[i, j])
                    if p >= self.LTP.shape[0]:
                        self.G[i, j] = self.LTD[-1]
                    else:
                        self.G[i, j] = self.LTD[p]


class Neurons:
    def __init__(self, neuron_number, batch_num, dt=1e-6):
        self.running_time = 1
        self.dt = dt  # 运行时间1 us
        self.neuron_number = neuron_number  # 神经元数目

        # 电路参数
        self.R1 = 1e4  # R1电阻为10kOhm
        self.R2 = 1e3  # R2阻值为100Ohm,R2是采样电阻,尽可能保证采样电阻和忆阻器的低组态在同一数量级
        self.c = 1e-9  # 电容为1nF

        # 忆阻器器件性能
        self.R_h = 1e6  # 忆阻器的高阻态（HRS）
        self.R_l = 1e3  # 忆阻器的低阻态（LRS）
        self.mean_th = 4.5  # 忆阻器阈值的均值
        self.mean_hold = 3.3  # 忆阻器保持电压的均值
        self.T_delay = 1e-6  # 忆阻器响应时间为1ms
        self.T_hold = 1e-6  # 忆阻器弛豫时间为1ms
        # 神经元参数
        self.V_th = self.mean_th * np.ones(self.neuron_number)  # 忆阻器阈值转变电压为4.5V；阈值电压的分布在运行中体现
        self.V_hold = self.mean_hold * np.ones(self.neuron_number)  # 忆阻器保持电压为0.05V
        self.R_m = self.R_h * np.ones(self.neuron_number)  # 忆阻器的阻值调到高阻态
        self.R_timer = -1 * np.ones(self.neuron_number)  # 记录忆阻器响应时间及弛豫时间
        self.th = 1 * np.ones(self.neuron_number)  # 神经元阈值：设置为0.001V
        self.f_output = np.zeros(self.neuron_number)  # 神经元输出频率初始化为0
        self.f_input = np.zeros(self.neuron_number)  # 输入频率初始化为0
        self.V_m = np.zeros(self.neuron_number)  # 神经元膜电位初始化为0
        self.V_s = np.zeros(self.neuron_number)  # 采样电阻分压
        self.firing = np.zeros(self.neuron_number)  # 记录点火信息
        self.Record_output = np.zeros((batch_num * 4, self.neuron_number))  # 神经元激发的情况
        self.output_timer = 0  # 定义一个计数器，用于计数每一个图片输入后神经元的激发情况
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
        self.spikes_num = 6
        self.spikes_V = 3
        self.input_spikes_num = np.zeros(self.neuron_number)  # 记录输入脉冲的数量

        # 设计一个state记录神经元的触发情况
        self.state = np.zeros(self.neuron_number)

    def run_neurons(self, input_spikes):
        # 根据输入信号确定时间步长,time_step即为输入脉冲的个数
        # 每一次输入都会将记录置为 0
        # 按时间步长开始运行
        # time_step = input_spikes.shape[0]
        time_step = self.spikes_num
        Record_V_m = np.zeros((time_step, self.neuron_number))
        Record_V_s = np.zeros((time_step, self.neuron_number))
        Record_V_o = np.zeros((time_step, self.neuron_number))
        Record_V_th = np.zeros((time_step, self.neuron_number))
        Record_R_m = np.zeros((time_step, self.neuron_number))

        output_spikes_num = np.zeros(self.neuron_number)  # 记录输出脉冲的数量

        # self.input_spikes_num = np.zeros(input_spikes.shape[1])                # 记录输入脉冲的数量
        # self.input_spikes_num[input_spikes[:, 1] != 0] = self.spikes_num

        # 每一回输入前将忆阻器置于高电导态
        self.R_m = self.R_h * np.ones(self.neuron_number)  # 忆阻器初始化为高电阻态
        self.V_m = np.zeros(self.neuron_number)  # 神经元膜电位初始化为 0
        self.V_s = np.zeros(self.neuron_number)  # 采样电压初始化为 0
        self.R_timer = -1 * np.ones(self.neuron_number)  # 记录忆阻器响应时间及弛豫时间初始化为-1
        self.f_output = np.zeros(self.neuron_number)  # 神经元输出频率初始化为0

        # 记录所有神经元的发放情况

        self.Record_output_spikes = np.zeros((time_step, self.neuron_number))

        # time_step决定了input_spikes将要输入几次
        for i in range(time_step):
            # 根据电路方程计算各参量
            I1 = (input_spikes - self.V_m) / self.R1
            I2 = self.V_m / (self.R_m + self.R2)
            du = (I1 - I2) * self.dt / self.c  # 电压的变化量
            du = du.reshape(self.neuron_number)  # 此处的reshape主要调整du的维度(1,4)为(,4)
            self.V_m += du
            self.V_s = self.V_m / (self.R_m + self.R2) * self.R2

            # 根据采样电阻分压确定神经元是否激发
            self.firing[self.V_s > self.th] = 1

            # 判断忆阻器是否发生阈值转变
            # 判断是否有多个神经元同时激发，如果有就选择电压信号最大的作为此次输出
            if np.sum(self.firing) > 1:
                det = self.V_s - self.th
                p = np.where(det == np.max(det))[0][0]
                self.firing = np.zeros(self.neuron_number)
                self.firing[p] = 1
                # self.f_output[p] += 1

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

                # # 判断是否点火并记录点火时间
                # # 寻找与阈值差距最大的神经元作为触发神经元
                # # 让其他没有触发的输出神经元的进入侧向抑制
                # if np.sum(self.firing) >= 1:
                #     det = self.V_s - self.th
                #     self.firing[det < np.max(det)] = 0
                #     output_spikes_num[self.firing == 1] += 1
                #     self.lateral_inhibition[self.firing == 0] = self.lateral_inhibition_time
                #
                # # 调整每一个输出神经元的阈值,判断是否进入侧向抑制
                # for neuron in range(self.neuron_number):
                #     if self.lateral_inhibition[neuron] == self.lateral_inhibition_time:
                #         self.th[neuron] = self.lateral_inhibition_th[0]
                #
                #     elif 0.6 * self.lateral_inhibition_time < self.lateral_inhibition[neuron] <= \
                #             0.8 * self.lateral_inhibition_time:
                #         self.th[neuron] = self.lateral_inhibition_th[1]
                #
                #     elif 0.4 * self.lateral_inhibition_time < self.lateral_inhibition[neuron] <= \
                #             0.6 * self.lateral_inhibition_time:
                #         self.th[neuron] = self.lateral_inhibition_th[2]
                #
                #     elif 0.2 * self.lateral_inhibition_time < self.lateral_inhibition[neuron] <= \
                #             0.4 * self.lateral_inhibition_time:
                #         self.th[neuron] = self.lateral_inhibition_th[3]
                #
                #     else:
                #         # 神经元阈值逐渐恢复为静息电压
                #         self.th[neuron] = 0.001
                #
                # self.lateral_inhibition -= 1
                # self.lateral_inhibition[self.lateral_inhibition < 0] = 0

            # 记录神经元各节点电压
            Record_V_m[i] = self.V_m
            Record_V_s[i] = self.V_s
            Record_V_o[i] = self.firing
            Record_R_m[i] = self.R_m
            Record_V_th[i] = self.th

            # 如果神经元激发，则选取与阈值电压差距最大的神经元记为输出神经元
            if np.sum(self.firing) >= 1:
                det = self.V_s - self.th
                self.firing[det < np.max(det)] = 0

                self.f_output[self.firing == 1] += 1  # 记录输出神经元的输出频率

                # 如果记录状态state所对应的神经元为0
                if self.state[self.firing == 1] == 0:
                    # 激发的神经元进入不应期
                    self.time_counter[self.firing == 1] = self.refractory_period_time

                    # 激发的神经元进入不应期后将其状态量调整为1
                    self.state[self.firing == 1] = 1
                    # 未激发的神经元进入侧向抑制
                    self.time_counter[self.state == 0] = self.lateral_inhibition_time

            # 使用SRM神经元点火后，在两个(固定)时间步长内应该输入不应期及侧向抑制，固定时间步长后应该恢复静息电位输入
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
                        self.th[j] = self.refractory_period_th[
                            int(len(self.refractory_period_th) - self.time_counter[j])]
                    else:
                        self.th[j] = self.lateral_inhibition_th[
                            int(len(self.refractory_period_th) - self.time_counter[j])]
                else:
                    self.th[j] = 0.05

            # 将计时器小于零的置为0
            self.time_counter -= 1
            self.time_counter[self.time_counter < 0] = 0

            self.firing = np.zeros(self.neuron_number)

        # # 统计四个神经元一共的激发脉冲数目
        # Spikes_num = np.sum(Record_V_o, axis=0)
        #
        # # 选出激发次数最多的神经元作为输出神经元
        # chose_neuron = np.argmax(Spikes_num)
        # self.Record_output[self.output_timer, chose_neuron] = 1
        # self.output_timer += 1
        #
        # # 将激发次数最多的神经元将进入不应期
        # self.time_counter[chose_neuron] = self.refractory_period_time

        # 可视化各节点电压的变化过程
        # input_spikes为输入电压，V_m为忆阻器分电压，
        # self.visualize(input_spikes, Record_V_m, Record_V_s, Record_V_th, Record_V_o, Record_R_m)
        # self.visualize(Record_V_m, Record_V_o)
        # self.visualize(Record_V_m, Record_V_s, Record_V_o, Record_V_th)

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
            plt.subplot(var_num, 1, i + 1)
            plt.plot(x, Var[i][:, 0])
            plt.xticks([])

        plt.show()
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

    def encoding(self, image):
        # 编码的意义在于将图像的像素信号转化为电压脉冲输入
        # 使用reshape函数将image转化为(n，1)
        picture = image.reshape((-1, 1))
        p_x, p_y = picture.shape
        # 设计一个空列表用来储存
        spikes = np.zeros((p_x, p_y))

        # 编码规则为如果二值化后的图像像素点为“1”则输出五个脉冲，脉冲为0.1V,如果像素点为0，则输出为0V
        for counter in range(p_x):
            if picture[counter] == 1:
                spikes[counter, 0] = self.spikes_V
            else:
                spikes[counter, 0] = 0

        return spikes

    def f_encoding(self, image):
        # f_encoding的意义在于计算输入的脉冲频率
        # reshape成n行1列的数据结构
        picture = image.reshape((-1, 1))
        p_x, p_y = picture.shape
        # 空列表储存spike
        spikes = np.zeros((p_x, p_y))

        # 编码规则为如果二值化后的图像像素点为“1”则输出五个脉冲，脉冲为0.1V,如果像素点为0，则输出为0V
        for counter in range(p_x):
            if picture[counter] == 1:
                spikes[counter, 0] = self.spikes_num
            else:
                spikes[counter, 0] = 0

        return spikes

    def SRM(self, Record_V_o):
        # 关于SRM神经元的一些重要的参数
        # 侧向抑制，暂时不用
        self.Lateral_inhibition = 5

        # 不应期，暂时不用
        self.Refractory_period = 5

        # 读出Record_V_o的时间步长
        time_step = Record_V_o.shape[0]

        # 神经元激发的情况,遍历Record_V_o,发现神经元被激发后，记录并退出循环，即记录第一个激发的神经元
        # 换一种SRM的神经元比较算法，即记录输出脉冲最多的数量
        # for i in range(time_step):
        #     if np.sum(Record_V_o[i, :]) == 1:
        #         self.Record_output[self.output_timer] = Record_V_o[i, :]
        #         break
        # self.output_timer += 1


class input_neurons:
    def __init__(self):  # 一种特解
        self.pixel_num = 5 * 5  # 输入神经元的个数
        self.dt = 1e-6  # 时间步长 1 us
        self.pulse_t = 2e-4  # 一个脉冲的持续时间1 ms
        self.pulse_num = 5  # 一次激发的脉冲数目
        self.V = 0.13  # 脉冲的电压值
        self.due = 0.5  # 脉冲的占空比为0.5
        self.frequency = 10  # 脉冲频率为10
        self.T = int(self.pulse_t / self.dt)  # 脉冲的周期为200
        # 计算一张图输入编码所需要的时间
        self.encode_time = int(self.pulse_num * self.T)  # 所有的脉冲所需要的编码时间

        # 制作一个空集合存储编码
        # 计算时间例如：batch: 20, spike长度为25*20
        self.spike = np.zeros((self.pixel_num, self.encode_time))  # 此处的输出脉冲应该为25 * 500

    def encoding(self, image):
        # 二值化图像，使用HUST的时候不需要
        # image = self.binary(image)

        # 将图像编码
        for p_n in range(self.pixel_num):
            if image[p_n] == 0:
                for i in range(self.encode_time):
                    self.spike[p_n, i] = 0
            elif image[p_n] == 1:
                for i in range(self.pulse_num):
                    for j in range(int(self.T * self.due)):
                        self.spike[p_n, i * self.T + j + 1] = self.V
        return self.spike

    def binary(self, image):

        # 二值化图像，灰度值大于200的置为1，灰度值小于200的置为0
        for counter in range(self.pixel_num):
            if image[counter] >= 200:
                image[counter] = 1
            else:
                image[counter] = 0

        return image


def encoding(image):
    # 使用reshape函数将image转化为(n，1)
    picture = image.reshape((-1, 1))

    # 一个空列表用以存储spike
    spike_train = []
    # 编码规则为二值化后的图像如果像素点为"1"，则代表五个连续的脉冲,每个脉冲的脉宽为50 ms，脉高为0.1V，如果像素点为“0”,则为0V
    for counter in range(len(picture)):

        # 每一个时间步长为10ms
        if picture[counter] == 0:
            for t_0 in range(25):
                spike_train.append(0)
        else:
            for t_1 in range(25):
                spike_train.append(0.1)

    spike_input_train = np.array(spike_train)

    return spike_input_train


def train_image(path='E:/新建文件夹/工作/SSI/gasdata/Project/hybirdtrainset/hybirdtrainset_0.xlsx'):
    img = np.delete(np.array(pd.read_excel(path)).T, 0, 0)
    '''
    f, ax = plt.subplot()
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img[i].reshape(5, 5))
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
    '''

    return img


def main():
    images = train_image()

    # 初始化神经元和突触
    Synapse = synapse(25, 4)               # 突触
    N = Neurons(4,80)                        # 输出神经元
    spike= input_neurons()                                          # 输入神经元

    for batch in range(7):
        plt.figure(batch)
        N.state = np.zeros(4)
        Synapse.w=np.zeros((25, 4))
        for j in range(4):
            plt.subplot(4,1,j+1)
            plt.imshow(np.reshape(Synapse.G[:, j], (5, 5)))

        plt.show()
        for i in range(40):
            # start_time=time.time()                                #没用上时间
            # input_v=N.encoding(images[i])                           #脉冲电压
            input_f=N.f_encoding(images[i])                         #频率
            out_spike=Synapse.VMM_operation(input_f)
            # 运行神经元
            N.run_neurons(out_spike)
            # 训练神经单元SRDP法则
            Synapse.SRDP(input_f, N.f_output)
            # end_time=time.time()
        #一个批次后进行权重更新，然后重置w
        # for j in range(4):
        #     plt.subplot(4, 1, j + 1)
        #     plt.imshow(np.reshape(Synapse.w[:, j], (5, 5)))
        # plt.show()
        Synapse.Update()

        # np.savetxt(r'E:/新建文件夹/工作/SSI/gasdata/Project/synapse_weight/synapse_G.txt', Synapse.G_num, fmt='%d',delimiter=" ")
        data = pd.DataFrame(Synapse.G)
        writer = pd.ExcelWriter(r'synapse_weight/synapse.xlsx')
        data.to_excel(writer)
        writer.close()
        # for j in range(4):
        #     plt.subplot(4, 1, j + 1)
        #     plt.imshow(np.reshape(Synapse.G[:, j], (5, 5)))
        # plt.show()
    # plt.imshow(N.Record_output)
    # plt.show()

if __name__=="__main__":
    main()

    '''
    epoch = 1                    # 训练的epoch
    batch = 20                    # 训练的batch
    labels, images = get_image()

    choose_num = 6000  # 指定一个编号，你可以修改这里

    # 初始化神经元和突触
    Synapse = synapse(784, 10)    # 突触`
    N = Neurons(10)               # 输出神经元
    Spike = input_neurons()       # 输入神经元

    for i in range(epoch):
        for j in range(batch):
            # 将图片一个个取出后进行编码
            input_v = Spike.encoding(images[epoch * 20 + batch], labels[epoch * 20 + batch])
            out_spike = Synapse.VMM_operation(input_v.T)

            # 运行神经元
            N.run_neurons(out_spike)

    plt.imshow(N.Record_output)
    plt.show()
    '''
    '''
    # 将所需要的图像取出
    image = images[6000]
    label = labels[6000]

    Synapse = synapse(784, 10)

    N = Neurons(10)

    Spike = input_neurons()
    input_v = Spike.encoding(image, label)

    out_spike = Synapse.VMM_operation(input_v.T)

    # 编码后的spike输入数量
    spike_num = 5
    N.run_neurons(out_spike)

    plt.imshow(out_spike)
    plt.show()
    # 一个epoch中有20个batch,而一个batch中有20个image
    '''
    '''
    for epoch in range(20):

        for batch in range(20):

            # 先将所需要的image 和 label取出
            image = images[epoch * 20 + batch]
            label = labels[epoch * 20 + batch]

            # 先将需要的图像编码
            Spike = input_neurons()
            input_v = Spike.encoding(image, label)

            # 调用师兄写的突触模拟
            Synapse = synapse(784, 10)
            out_spike = Synapse.VMM_operation(input_v)
            N = Neurons(10)
            N.run_neurons(out_spike)

    # plt.imshow(out_spike)
    # plt.show()
    '''
    '''plt.imshow(input_spike)

    plt.show()
    '''
    '''
    label = labels[choose_num]

    image = images[choose_num].reshape(28,28)

    # 二值化处理输入样本
    '''
    # 一个训练周期有20个样本
    '''for batch in range(20):

        picture = images[batch]

        for encoding_num in range(28*28):
            input_neurons
    '''

    '''spike_train = encoding(image)

    plt.title('the label is : {}'.format(label))
    plt.imshow(image)
    time = np.arange(len(spike_train))
    plt.plot(time, spike_train)
    plt.show()
    '''

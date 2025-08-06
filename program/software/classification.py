import time
import numpy as np
import struct
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.nn
import os


def get_feature(path='synapse_weight/feature.xlsx'):
    feature = np.delete(np.array(pd.read_excel(path)).T, 0, 0)
    return feature


def standard_spike(feature, kernel_size=2, padding=0, stride=1):  # 根据标准的feature进行了一次cnn
    standard1 = np.zeros((4, 16))
    standard2 = np.zeros((4, 16))
    feature_num = feature.shape[0]
    feature_size = int(np.sqrt(feature.shape[1]))
    output_size = int((feature_size - kernel_size + padding) / stride + 1)
    for num in range(feature_num):
        for line in range(output_size):
            standard1[num][line] = feature[num][line] + feature[num][line + 1] + feature[num][line + 5] + feature[num][
                line + 6]  # 这里是针对2×2卷积核权重全为1的特解
            standard1[num][line + 4] = feature[num][line + 5] + feature[num][line + 6] + feature[num][line + 10] + \
                                       feature[num][line + 11]
            standard1[num][line + 8] = feature[num][line + 10] + feature[num][line + 11] + feature[num][line + 15] + \
                                       feature[num][line + 16]
            standard1[num][line + 12] = feature[num][line + 15] + feature[num][line + 16] + feature[num][line + 20] + \
                                        feature[num][line + 21]
    # for num in range(feature_num):
    #     for n in range(output_size):
    #         for line in range(output_size):
    #             for i in range(kernel_size):
    #                 standard2[num][line+n*4]+=feature[num][line+i+n*5]+feature[num][line+i+feature_size+n*5]#其实这也是特解
    return standard1


def standard_map(spike):
    m1 = np.zeros((4, 16))  # 第一特征，即小于等于1
    m2 = np.zeros((4, 16))  # 第二特征，即大于等于3
    num = spike.shape[0]
    for row in range(num):
        for line in range(16):
            if spike[row][line] <= 1:
                m1[row][line] = 1
            elif spike[row][line] >= 3:
                m2[row][line] = 1
    return m1, m2


# 我真是该死为什么要用类呢
# 这一段本来有快一百行代码，后来全删了


def get_spike(path='hybirdtrainset/hybirdtrainset_0.xlsx'):
    spike = np.delete(np.array(pd.read_excel(path)).T, 0, 0)
    return spike


def normal_spike(feature):  # 这里的卷积操作仍然是特解
    normal = np.zeros(16)  # 初始化输出

    for line in range(4):
        normal[line] = feature[line] + feature[line + 1] + feature[line + 5] + feature[line + 6]  # 这里是针对2×2卷积核权重全为1的特解
        normal[line + 4] = feature[line + 5] + feature[line + 6] + feature[line + 10] + feature[line + 11]
        normal[line + 8] = feature[line + 10] + feature[line + 11] + feature[line + 15] + feature[line + 16]
        normal[line + 12] = feature[line + 15] + feature[line + 16] + feature[line + 20] + feature[line + 21]

    return normal


def normal_map(spike):
    m1 = np.zeros(16)  # 第一特征，即小于等于1
    m2 = np.zeros(16)  # 第二特征，即大于等于3

    for line in range(16):
        if spike[line] <= 1:
            m1[line] = 1
        elif spike[line] >= 3:
            m2[line] = 1
    return m1, m2


def sum_shuffle(end: int, start: int = 0):
    # 将多个文件中的数据总和起来并随机打乱,end表示最后一个文件数，start表示第一个文件数
    # 默认start是0，end最多到15
    f_set = np.zeros(((end - start + 1) * 40, 26))

    for num in range(start, end + 1):
        path = 'hybirdtrainset/hybird_label_%d.xlsx' % num
        set = get_spike(path)
        for n in range(40):
            f_set[n + num * 40] = set[n]
    np.random.shuffle(f_set)

    return f_set


def main():
    # 计算出标准特征值
    # feature=get_feature()
    # print(feature)
    # spike=standard_spike(feature)
    # print(spike)
    # w1,w2=standard_map(spike)
    # 发现通过标准特征值得到的识别准确率低得吓人

    # --------------------这一段仅用来测试----------------
    # print(w1,'\n',w2)
    # test1=np.array((0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0))
    # test2=np.array((0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1))
    # score1=np.zeros(4)
    # score2=np.zeros(4)
    # score=np.zeros(4)
    # for num in range(4):
    #     for i in range(16):
    #         if test1[i]==w1[num][i]==1:
    #             score1[num]+=1
    #         if test2[i]==w2[num][i]==1:
    #             score2[num]+=1
    # score=score1+score2
    # print(score)
    # --------------------测试结束，希望最终结果也跟测试结果一样----------------
    # 结果不顺利，最后自己写了一个训练程序，然后准确率就飙升了

    '''
    #发现这一段的代码不管用
    set=get_spike()
    correct=0
    for n in range(40):
    # n=1
        test_spike=normal_spike(set[n])
        m1,m2=normal_map(test_spike)
        # print(m1,'\n',w1[1],'\n',w1[2],'\n',m2,'\n',w2[1],'\n',w2[2])
        neuron_1=np.zeros(4)
        neuron_2=np.zeros(4)
        neuron=np.zeros(4)
        for num in range(4):
            for i in range(16):
                if m1[i]==w1[num][i]:
                    neuron_1[num]+=1
                if m2[i]==w2[num][i]:
                    neuron_2[num]+=1
        x=0.45
        neuron=x*neuron_1+(1-x)*neuron_2
        print('neuron_1=',neuron_1)
        print('neuron_2=',neuron_2)
        print('neuron=', neuron)
        print(int(n/10),np.where(np.max(neuron)==neuron)[0])
        if int(n/10)==np.where(np.max(neuron)==neuron)[0][0]:
            correct+=1
    accuracy=correct/40
    print('accuracy=',accuracy)
    '''
    # 初始化两个特征向量
    t1 = np.random.random((4, 16))
    t2 = np.random.random((4, 16))
    rate = 0.0004  # 可以把这个叫做学习率
    weight = 0.43  # 决定哪组神经元的比重会大一些
    acc = []
    t_acc=[]
    max_acc=0
    set = sum_shuffle(3)
    # rate_decline=0
    for epoch in range(500):
        # rate/=(epoch+1)
        # rate_decline+=1
        # if rate_decline==60:
        #     rate/=2
        #     rate_decline=0
        if epoch ==50:
            rate/=8
        elif epoch ==75:
            rate/=4
        elif epoch == 100:
            rate/=4
        elif epoch == 200:
            rate /= 2
        elif epoch == 300:
            rate /= 2
        elif epoch == 400:
            rate /= 2

        x1 = np.zeros((4, 16))
        x2 = np.zeros((4, 16))

        # for step in range(1):
        correct = 0
        for n in range(120):#训练集
            # n=1
            x1[t1 >= 0.5] = 1
            x1[t1 < 0.5] = 0
            x2[t2 >= 0.5] = 1
            x2[t2 < 0.5] = 0
            test_spike = normal_spike(set[n][:-1])
            m1, m2 = normal_map(test_spike)
            neuron_1 = np.zeros(4)
            neuron_2 = np.zeros(4)

            # 识别过程
            for num in range(4):
                for i in range(16):
                    if m1[i] == x1[num][i]:
                        neuron_1[num] += 1
                    if m2[i] == x2[num][i]:
                        neuron_2[num] += 1
            x = weight
            neuron = x * neuron_1 + (1 - x) * neuron_2
            result = np.where(np.max(neuron) == neuron)[0][0]
            label = int(set[n][-1])

            # 训练过程
            if label == result:
                correct += 1  # 正确结果加一个
                for i in range(16):  # 识别正确则增强对应的特征向量，不惩罚其他的特征向量

                    # 这段代码写完发现其实可以精简，但是也不是不能用。而且既然能用了我为什么要为了运行快那么几毫秒去改?
                    # 强迫症犯了，还是改了

                    # 第一特征
                    if m1[i] == 1 and t1[result][i] <= 1 - rate/2:
                        t1[result][i] += rate/2
                    elif m1[i] == 0 and t1[result][i] >= rate/2:
                        t1[result][i] -= rate/2
                    # 第二特征
                    if m2[i]  == 1 and t2[result][i] <= 1 - rate/2:
                        t2[result][i] += rate/2
                    elif m2[i]  == 0 and t2[result][i] >= rate/2:
                        t2[result][i] -= rate/2

            else:  # 识别出错则增强正确的特征向量，并惩罚出错的向量
                for i in range(16):

                    # 增强
                    # 第一特征
                    if m1[i] == 1 and t1[label][i] <= 1 - rate :
                        t1[label][i] += rate
                    elif m1[i]  == 0 and t1[label][i] >= rate :
                        t1[label][i] -= rate
                    #第二特征
                    if m2[i]  == 1 and t2[label][i] <= 1 - rate :
                        t2[label][i] += rate
                    elif m2[i]  == 0 and t2[label][i] >= rate :
                        t2[label][i] -= rate

                    # 惩罚
                    # 第一特征
                    if m1[i] == 1 and t1[result][i]>= rate/2:
                        t1[result][i] -= rate / 2
                    elif m1[i] == 0 and t1[result][i]<= 1-rate/2:
                        t1[result][i] += rate / 2
                    # 第二特征
                    if m2[i] == 1 and t1[result][i]>= rate/2:
                        t2[result][i] -= rate / 2
                    elif m2[i] == 0 and t1[result][i]<= 1-rate/2:
                        t2[result][i] += rate / 2
        ct=0
        for n in range(40):#测试集

            x1[t1 >= 0.5] = 1
            x1[t1 < 0.5] = 0
            x2[t2 >= 0.5] = 1
            x2[t2 < 0.5] = 0
            test_spike = normal_spike(set[n+120][:-1])
            m1, m2 = normal_map(test_spike)
            neuron_1 = np.zeros(4)
            neuron_2 = np.zeros(4)

            # 识别过程
            for num in range(4):
                for i in range(16):
                    if m1[i] == x1[num][i]:
                        neuron_1[num] += 1
                    if m2[i] == x2[num][i]:
                        neuron_2[num] += 1
            x = weight
            neuron = x * neuron_1 + (1 - x) * neuron_2
            result = np.where(np.max(neuron) == neuron)[0][0]
            label = int(set[n+120][-1])

            if label == result:
                ct+=1

        accuracy = correct / 120
        t_accuracy = ct/40
        t_acc.append(t_accuracy)
        acc.append(accuracy)
        print('epoch=', epoch + 1, 'accuracy=%.4f' % accuracy,'t_accuracy=%.4f' %t_accuracy)
        if accuracy>=max_acc:
            max_acc = accuracy
            max_epoch = epoch+1
            m1=x1
            m2=x2
    print('max_acc=%.4f'%max_acc,'epoch=%d'%int(max_epoch))
    x = [i for i in range(1, 501)]
    # 保存特征向量和准确率数据
    data = {
        'f1': list(x1),
        'f2': list(x2)
    }
    df = pd.DataFrame(data)
    df.to_excel('classification/clas_feature4.xlsx', index=False)
    # 最准特征向量
    data = {
        'f1': list(m1),
        'f2': list(m2)
    }
    df = pd.DataFrame(data)
    df.to_excel('classification/max_feature4.xlsx', index=False)
    #训练准确率
    data2 = {
        'epoch': list(x),
        'accuracy': list(acc)
    }
    df2 = pd.DataFrame(data2)
    df2.to_excel('classification/accuracy4.xlsx', index=False)
    #测试准确率
    data3 = {
        'epoch': list(x),
        'accuracy': list(t_acc)
    }
    df3 = pd.DataFrame(data3)
    df3.to_excel('classification/t_accuracy4.xlsx', index=False)
    # 绘制准确率曲线

    plt.figure()

    # 绘制折线图
    plt.plot(x, acc, marker='o', linestyle='-', color='b', label='Accuracy')

    # 设置标题和标签
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

    plt.plot(x, t_acc, marker='o', linestyle='-', color='b', label='Accuracy')

    # 设置标题和标签
    plt.title('Model t_Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

if __name__ == "__main__":
    main()

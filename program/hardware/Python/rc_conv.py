# %% RC卷积
import os
import time
import numpy as np
import pandas as pd
from wr_ctrl import run
from conv_util import get_vmem
# %% 输入脉冲
def get_input():
    file_list = ['hybird_label_0', 'hybird_label_1', 'hybird_label_2', 'hybird_label_3']
    data, label = [], []
    for fname in file_list:
        f_path = f'./data/input_rc/{fname}.xlsx'
        df = pd.read_excel(f_path, header=0).to_numpy()[:,1:]
        for i in range(df.shape[1]):
            data.append(df[:-1,i])
            label.append(df[-1,i])
    data = np.array(data).reshape(-1,5,5)
    label = np.array(label)
    np.savez(f'./data/input_rc/code.npz', data=data, label=label)
# %% 卷积配置
def conf_conv(t, h, w):
    conf = {}
    conf['wl'], conf['bl'] = [data[t,h,w],data[t,h,w+1],data[t,h+1,w],data[t,h+1,w+1]], [1, 1, 1, 1]
    conf['ad_num'] = 4
    conf['ad_chan'] = '00101010101'
    conf['sl_amp'] = 3
    conf['da_amp'], conf['A_in'] = -0.2, 1
    conf['da_wid'], conf['ad_dly'] = 2e-3, 0.5e-3
    return conf
# %% 卷积运算
def run_conv(data, fname):
    N, H, W = data.shape
    data_out = np.zeros((N, W-1, H-1))
    for n in range(N):
        for h in range(H-1):
            for w in range(W-1):
                conf = conf_conv(n, h, w)
                if conf['wl'] == [0, 0, 0, 0]:
                    I_mem = [0, 0, 0, 0]
                else:
                    I_mem = run('c', conf, fname)
                    time.sleep(conf['da_wid'])
                print(f'n{n}_h{h}_w{w}: {I_mem[0]}')
                data_out[n, h, w] = I_mem[0]
    return data_out
# %% 主程序
if __name__ == '__main__':
    # get_input()
    # 输入
    data = np.load(f'./data/input_rc/code.npz')['data']
    label = np.load(f'./data/input_rc/code.npz')['label']
    print(f'data: {data.shape}', f'label: {label.shape}')
    # data = data[:10]
    # label = label[:10]
    # # 运行
    # fname = 'rc'
    # data_out = run_conv(data, fname)
    # np.savez(f'./data/{fname}_out.npz', data=data_out)
    # print(f'data_out: {data_out.shape}')
    # np.savetxt(f'./data/{fname}_out.csv', data_out.reshape(data.shape[0], -1), delimiter=',')
    # 输出
    fname = 'rc'
    data_out = np.load(f'./data/{fname}_out.npz')['data']
    print(f'max: {data_out.max()}, min: {data_out.min()}')
    vmem = get_vmem(data_out, i_high=120, i_low=70, val_A=43, val_B=57)
    np.savez(f'./data/{fname}_vmem.npz', data=vmem)
    np.savetxt(f'./data/{fname}_vmem.csv', vmem.reshape(data.shape[0], -1), delimiter=',')
    # 结果
    pred = vmem.argmax(axis=1)
    acc = np.sum(pred==label)/len(label)
    print(f'acc: {acc}')
    result = np.zeros((label.shape[0], 2))
    result[:,0], result[:,1] = label, pred
    np.savetxt(f'./data/{fname}_result.csv', result, delimiter=',')

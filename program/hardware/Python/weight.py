#%% 权重调节
import os
import time
import pandas as pd
from wr_ctrl import run
#%% 通道映射
def chan_map(bl_loc):
    chan_dist = {1 : '00000000100',
                 2 : '00000010000',
                 3 : '00001000000',
                 4 : '00100000000'}
    return chan_dist[bl_loc]
#%% 读
def read(R_conf):
    conf = {}
    loc = R_conf['loc']
    conf['loc'] = loc
    conf['wl'], conf['bl'] = [0, 0, 0, 0], [0, 0, 0, 0]
    conf['wl'][4-loc[0]], conf['bl'][4-loc[1]] = 1, 1
    conf['ad_num'] = R_conf['ad_num']
    conf['ad_chan'] = chan_map(loc[1])
    conf['sl_amp'] = 3
    conf['da_amp'], conf['A_in'] = -0.2, 1
    conf['da_wid'], conf['ad_dly'] = R_conf['da_wid'], R_conf['ad_dly']
    return conf
#%% 写
def write(W_conf):
    conf = {}
    loc = W_conf['loc']
    conf['loc'] = loc
    conf['wl'], conf['bl'] = [0, 0, 0, 0], [0, 0, 0, 0]
    conf['wl'][4-loc[0]], conf['bl'][4-loc[1]] = 1, 1
    conf['ad_num'] = 0
    conf['ad_chan'] = '000000' + '0000' + '0'
    conf['sl_amp'] = W_conf['sl_amp']
    conf['da_amp'], conf['A_in'] = W_conf['da_amp'], 1
    conf['da_wid'], conf['ad_dly'] = W_conf['da_wid'], 3e-6
    return conf
#%% 早停
def stop(mode_seq, R_mem, R_set):
    if R_mem <= 0:
        return False
    if ('p' in mode_seq) and (R_mem < R_set):
        return True
    elif ('n' in mode_seq) and ((R_mem > R_set) or (R_mem < 0)):
        return True
    else:
        return False
#%% 运行程序
def run_seq(fname, mode_seq, R_conf, P_conf, N_conf, R_set, note, R_num=1):
    # 读取序号
    if not os.path.exists(f'./data/{fname}.csv'):
        NO = 1
    else:
        df = pd.read_csv(f'./data/{fname}.csv', encoding='gbk')
        NO = int(df.tail(1)['NO']+1)
    # 运行序列
    val_p, val_n = P_conf['val'], N_conf['val']
    for mode in mode_seq:
        if mode == 'R':
            conf = read(R_conf)
            conf['NO'] = NO
            conf['note'] = f'R{R_num}_r{conf["ad_num"]}'
            R_mem = 0
            for i in range(R_num):
                R_mem += run(mode, conf, fname)
                time.sleep(conf['da_wid'])
            R_mem /= R_num
        else:
            if mode == 'r':
                conf = read(R_conf)
            elif mode == 'p':
                conf = write(P_conf)
                P_conf[val_p] += P_conf['step']
            elif mode == 'n':
                conf = write(N_conf)
                N_conf[val_n] += N_conf['step']
            else:
                raise ValueError('Invalid')
            conf['NO'] = NO
            conf['note'] = note
            R_mem = run(mode, conf, fname)
            time.sleep(conf['da_wid'])
            time.sleep(100e-3)
        # 早停
        if (mode == 'r') and stop(mode_seq, R_mem, R_set):
            print('Stop')
            G_mem = 1/R_mem
            return G_mem, True
    G_mem = 1/R_mem
    return G_mem, False
#%% 主程序
if __name__ == '__main__':
    fname = 'RRR'
    loc = (4, 1)
    mode_seq = 'R' + 'R'*9
    # mode_seq = 'r'*5 + 'pr'*10 + 'r'*5
    R_conf = {'loc':loc, 'ad_num':4, 'da_wid':1e-3, 'ad_dly':0.5e-3}
    P_conf = {'loc':loc, 'val':'da_amp', 'sl_amp':3, 'da_amp':1, 'da_wid':10e-6, 'step':0.05}
    N_conf = {'loc':loc, 'val':'da_amp', 'sl_amp':3, 'da_amp':-1, 'da_wid':10e-6, 'step':-0.02}
    R_set = 1/1.2
    note = '4'
    run_seq(fname, mode_seq, R_conf, P_conf, N_conf, R_set, note, R_num=1)

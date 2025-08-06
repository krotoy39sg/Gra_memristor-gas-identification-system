#%% 权重自动调控
import time
import numpy as np
from weight import run_seq
#%% 计算下一电导态
def get_G_set(G_mem, G_idx, G_array):
    # 两端
    if G_mem < G_array[0]:
        return 'set', G_array[0]
    elif G_mem > G_array[-1]:
        return 'reset', G_array[-1]
    # 中间
    for i in range(G_array.size):
        if G_mem == G_array[i]:
            if i < G_idx:
                return 'set', G_array[i+1]
            else:
                return 'reset', G_array[i-1]
        elif G_mem < G_array[i]:
            if i <= G_idx:
                return 'set', G_array[i]
            else:
                return 'reset', G_array[i-1]
            
#%% 运行程序
def auto_run(fname, P_conf, N_conf, G_conf, G_idx, note):
    G_array = np.arange(G_conf['min'], G_conf['max'], G_conf['step'])
    G_t_max = G_array[G_idx] + G_conf['range']/2
    G_t_min = G_array[G_idx] - G_conf['range']/2
    val_p, val_n = P_conf['val'], N_conf['val']
    val_op, val_on = P_conf[val_p], N_conf[val_n]
    idx_o = G_conf['loc']
    while True:
        # 读取电导
        G_mem, mem_num = 0, 5
        for i in range(mem_num):
            time.sleep(1)
            mode, note = 'r', 'Read'
            R_set = 1 / G_array[G_idx]
            G_mem_temp, _ = run_seq(fname, mode, G_conf, P_conf, N_conf, R_set, note)
            G_mem += G_mem_temp
        G_mem /= mem_num
        print(f'G_mem = {G_mem:.3f} mS')
        # 调节成功
        if (G_mem <= G_t_max) and (G_mem >= G_t_min):
            print('success')
            # # 全局读取
            # for wl in range(4):
            #     for bl in range(1):
            #         G_conf['loc'] = (wl+1, bl+1)
            #         G_mem, mem_num = 0, 5
            #         for i in range(mem_num):
            #             time.sleep(1)
            #             mode, note = 'r', 'Rglobal'
            #             R_set = 1 / G_array[G_idx]
            #             G_mem_temp, _ = run_seq(fname, mode, G_conf, P_conf, N_conf, R_set, note)
            #             G_mem += G_mem_temp
            #         G_mem /= mem_num
            #         print(f'{G_conf["loc"]}: {G_mem:.3f} mS')
            # G_conf['loc'] = idx_o
            # # 电导保持性
            # mem_num = 100
            # for i in range(mem_num):
            #     time.sleep(1)
            #     mode, note = 'r', 'Retention'
            #     R_set = 1 / G_array[G_idx]
            #     run_seq(fname, mode, G_conf, P_conf, N_conf, R_set, note)
            return True
        # 更新电导
        next_opa, G_set = get_G_set(G_mem, G_idx, G_array)
        val_name = val_p if next_opa == 'set' else val_n
        val_data = P_conf[val_p] if next_opa == 'set' else N_conf[val_n]
        if next_opa == 'set':
            mode = 'r' + 'pr' * N_conf['num']
            # G_set = min(G_set, G_t_min)
            G_set = min(G_set, G_t_min) + G_conf['range']/4
        else:
            mode = 'r' + 'nr' * P_conf['num']
            # G_set = max(G_set, G_t_max)
            G_set = max(G_set, G_t_max) - G_conf['range']/4
        R_set = 1 / G_set
        if 'amp' in val_name:
            print(f'{next_opa} to {G_set:.3f} mS from {val_name} {val_data:.2f} V')
        elif 'wid' in val_name:
            print(f'{next_opa} to {G_set:.3f} mS from {val_name} {val_data*1e6:.0f} us')
        G_mem, G_get = run_seq(fname, mode, G_conf, P_conf, N_conf, R_set, next_opa)
        # 校验电导
        if G_get:
            time.sleep(1)
            mode = 'r' * 5
            R_set = 1 / G_array[G_idx]
            G_mem, _ = run_seq(fname, mode, G_conf, P_conf, N_conf, R_set, note)
            # 更新电压
            if ((next_opa == 'set') and (G_mem >= G_set)) or ((next_opa == 'reset') and (G_mem <= G_set)):
                P_conf[val_p], N_conf[val_n] = val_op, val_on
            else:
                if next_opa == 'set':
                    P_conf[val_p] += P_conf['step']
                else:
                    N_conf[val_n] += N_conf['step'] 
        else:
            if next_opa == 'set':
                P_conf[val_p] -= P_conf['dec']
            else:
                N_conf[val_n] -= N_conf['dec']
        # 终止条件
        if (P_conf[val_p]+P_conf['step']*P_conf['num'] > P_conf['max']) or\
            (abs(N_conf[val_n]+N_conf['step']*N_conf['num']) > abs(N_conf['max'])):
            print('fail')
            return False

#%% 主程序
def run(G_idx = 3):
    fname = '4cct0'
    note = '4cct0'
    loc = (1, 1)
    # 电导设置
    # G_conf = {'loc':loc, 'min':0.4, 'max':2.0, 'step':0.2, 'range':0.1,
    #           'ad_num':4, 'da_wid':1e-3, 'ad_dly':0.5e-3}
    G_conf = {'loc':loc, 'min':0.4, 'max':1.2, 'step':0.2, 'range':0.1,
            'ad_num':4, 'da_wid':1e-3, 'ad_dly':0.5e-3}
    # G_conf = {'loc':(1, 1), 'min':0.2, 'max':3.4, 'step':0.4, 'range':0.2,
    #           'ad_num':4, 'da_wid':1e-3, 'ad_dly':0.5e-3}
    # 电压扫描
    P_conf = {'loc':loc, 'sl_amp':3, 'da_amp':0.5, 'da_wid':1000e-6,
              'val':'da_amp', 'step':0.02, 'num':50, 'max':4.1, 'dec':0}
    N_conf = {'loc':loc, 'sl_amp':3, 'da_amp':-1.0, 'da_wid':1000e-6,
              'val':'da_amp', 'step':-0.02, 'num':50, 'max':-4.1, 'dec':-0}
    # # 脉宽扫描
    # P_conf = {'loc':(1, 1), 'sl_amp':3, 'da_amp':1, 'da_wid':1e-6,
    #           'val':'da_wid', 'step':2e-6, 'num':50, 'max':1e-4, 'dec':0}
    # N_conf = {'loc':(1, 1), 'sl_amp':3, 'da_amp':-2, 'da_wid':1e-6,
    #           'val':'da_wid', 'step':2e-6, 'num':50, 'max':1e-4, 'dec':0}
    # # 栅压扫描
    # P_conf = {'loc':(1, 1), 'sl_amp':0, 'da_amp':1, 'da_wid':5e-6,
    #           'val':'sl_amp', 'step':0.02, 'num':50, 'max':4.1, 'dec':0}
    # N_conf = {'loc':(1, 1), 'sl_amp':0, 'da_amp':-2, 'da_wid':10e-6,
    #           'val':'sl_amp', 'step':0.02, 'num':50, 'max':4.1, 'dec':0}
    result = auto_run(fname, P_conf, N_conf, G_conf, G_idx, note)
    return result

#%% 扫描
def scan():
    # for i in [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0]:
    for i in [0, 1, 2, 3, 2, 1, 0]:
        result = False
        while not result:
            # 电导设置
            G_idx = i
            result = run(G_idx)
            print(f'G_idx = {i}, result = {result}')
            if not result:
                pause = input()

if __name__ == '__main__':
    run()
    # scan()



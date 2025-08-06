##############################读写电路控制##############################
import os
import time
import pandas as pd
from SerialPort import SerialPort
from SerialPort import cut
##############################发送数据编码##############################
def encode(conf_dist):
    data = ''
    # 栅压控制
    data += '0000'
    data += f'{int(abs(conf_dist["sl_amp"]) * 1000) :012b}'
    # 开关控制
    data += '0000000'
    for b in conf_dist["bl"]:
        data += '1' if b==1 else '0'
    for w in conf_dist["wl"]:
        data += '1' if w==1 else '0'
    data += '1' if (conf_dist["da_amp"] < 0) else '0'
    # ADC控制
    data += f'{int(conf_dist["ad_num"]) :05b}'
    data += conf_dist["ad_chan"]
    data += f'{int(conf_dist["ad_dly"] * 1e6) :016b}'
    # DAC控制
    data += '000'
    data += '0' if (conf_dist["da_amp"] > 0) else '1'
    data += f'{int(abs(conf_dist["da_amp"]) * 1000 / conf_dist["A_in"]) :012b}'
    data += f'{int(conf_dist["da_wid"] * 1e6) :032b}'
    return data
##############################接收数据解码##############################
def decode(data):
    try:
        data_list = cut(data, 16)
        data_out = []
        for data in data_list:
            chan = int(data[0:4], 2)
            volt = int(data[4:16], 2)
            data_out.append([chan, volt])
        return data_out
    except Exception as err:
        print('Error:' + str(err))
        return None
##############################配置程序##############################
def conf(data_in, rec_len):
    try:
        ser = SerialPort()
        ser.report = False
        ser.baudrate = 115200
        ser.com = 'COM4'
        ser.port_open()
        if ser.isOpen():
            ser.send(data_in)
            if rec_len != 0:
                data_out = ser.receive(rec_len*2)
                data_out = decode(data_out)
            else:
                data_out = []
            ser.close()
            return data_out
    except Exception as err:
        print(err)
        ser.close()
##############################记录结果##############################
def record(mode, conf_dist, data_out, fname):
    # 电阻
    if mode in ['r', 'R']:
        data_out =  int(data_out[0][1]) * 5.0 / 4096
        Rs, A = 0.2, 220 / 22
        # R = (-conf_dist['da_amp']*conf_dist['A_in']* A/data_out[0][1]-1)*Rs\
        #     if data_out[0][1] else -1
        R = (-conf_dist['da_amp']*conf_dist['A_in']* A/(data_out+1e-6)-1)*Rs
        print(f'R{conf_dist["loc"]}:{R:.3f}K, {1/R:.3f}m\n')
    elif mode in ['c']:
        Rs, A = 0.2, 220 / 22
        R = [(data_out[1][1]-data_out[0][1]),
             (data_out[2][1]-data_out[0][1]),
             (data_out[3][1]-data_out[0][1]),
             (data_out[4][1]-data_out[0][1])]
    else:
        data_out = [['','']]
        R = ''
    # 保存
    if mode in ['r', 'R']:
        data = [conf_dist['NO'], mode, conf_dist['loc'], R,
                conf_dist['da_amp'], conf_dist['da_wid'],
                conf_dist['sl_amp'], time.strftime("%Y%m%d_%H-%M-%S"),
                data_out, conf_dist['note']]
        df = pd.DataFrame(data).T
        if not os.path.exists(f'./data/{fname}.csv'):
            header = ['NO', 'mode', 'loc', 'R', 'amp',
                    'wid', 'vg', 'time', 'vol', 'note']
            df.columns = header
            df.to_csv(f'./data/{fname}.csv', mode='a', encoding='gbk', index=None)
        else:
            df.to_csv(f'./data/{fname}.csv', mode='a', encoding='gbk', index=None, header=None)
    return R
##############################运行程序##############################
def run(mode, conf_dist, fname):
    rec_len = conf_dist['ad_chan'].count('1')
    data_in = encode(conf_dist)
    data_out = conf(data_in, rec_len)
    result = record(mode, conf_dist, data_out, fname)
    if mode in ['r', 'R', 'c']:
        return result
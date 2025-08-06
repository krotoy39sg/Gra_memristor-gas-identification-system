import serial
import serial.tools.list_ports
## 串口收发类
class SerialPort(serial.Serial):
    # ------------------------------------ 初始化 ----------------------------------- #
    def __init__(self, baudrate=9600, timeout=0.5, report=True, com=''):
        super().__init__()
        self.baudrate = baudrate
        self.timeout = timeout
        self.report = report
        self.com = com
    # ----------------------------------- 串口检测 ----------------------------------- #
    def port_detect(self):
        try:
            print('-----Port Detect-----')
            port_list = list(serial.tools.list_ports.comports())
            for port in port_list:
                print(str(port[0]))
        except Exception as err:
            print('Error:' + str(err))
            return None
    # ----------------------------------- 打开串口 ----------------------------------- #
    def port_open(self):
        try:
            if self.report:
                self.port_detect()
            if self.com == '':
                self.port = input('Port:')
            else:
                self.port = self.com
            self.open()
            # 判断串口的打开状态
            if self.isOpen() and self.report:
                print('-----Port Open-----')
        except Exception as err:
            print('Error:' + str(err))
            return None
    # ----------------------------------- 关闭串口 ----------------------------------- #
    def port_close(self):
        try:
            self.close()
            print('Port Close')
        except Exception as err:
            print('Error:' + str(err))
            return None
    # ----------------------------------- 发送数据 ----------------------------------- #
    def send(self, data_bin):
        try:
            if data_bin != None:
                data_list = cut(data_bin, 8)
                send_list = []
                hex_list = []
                for data in data_list:
                    data = int(data, 2)
                    send_list.append(data)
                    hex_list.append('{:02X}'.format(data))
                send_content = ' '.join(hex_list)
                send_byte = bytes(send_list)
                result = self.write(send_byte)
                if (self.report) and (result != 0):
                    print('Send--->' + send_content)
        except Exception as err:
            print('Error:' + str(err))
            return None
    # ----------------------------------- 接收数据 ----------------------------------- #
    def receive(self, length):
        try:
            while(True):
                num = self.inWaiting() # 接收缓存中的字节数
                if num >= length:
                    data_list = self.read(length)
                    bin_list = []
                    hex_list = []
                    for data in data_list:
                        bin_list.append('{:08b}'.format(data))
                        hex_list.append('{:02X}'.format(data))
                    data_bin = ''.join(bin_list)
                    receive_content = ' '.join(hex_list)
                    if self.report:
                        print('Receive<---' + receive_content)
                    return data_bin
        except Exception as err:
            print('Error:' + str(err))
            return None

## 切分数据
def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]

## 发送数据编码
def encode():
    try:
        data = ''
        # 时间
        times = input('t1, t2, t3 (us):\n').split(',')
        for i in range(len(times)):
            data += '{:016b}'.format(round(eval(times[i])))
        # 幅值
        amps = input('amps (V):\n').split(',')
        for i in range(len(amps)):
            data += '{:012b}'.format(round(eval(amps[i])*1000))
        return data
    except Exception as err:
        print('Error:' + str(err))
        return None
    
## 接收数据解码
def decode(data):
    try:
        data_list = cut(data, 12)
        print('V_measure:')
        for data in data_list:
            data = int(data, 2) * 5.0 / 4096
            print('{:.2f}V'.format(data), end=' ')
        print('')
    except Exception as err:
        print('Error:' + str(err))
        return None

## 主程序
if __name__ == '__main__':
    try:
        ser = SerialPort()
        while(True):
            if ser.isOpen():
                if input('Close?') == 'y':
                    ser.close()
                else:
                    data_out = encode()
                    ser.send(data_out)
                    data_in = ser.receive(6)
                    decode(data_in)
            else:
                if input('Open?') == 'y':
                    ser.port_open()
    except Exception as err:
        print(err)

import os
import serial
import pickle
import time
import argparse
from datetime import datetime

class ListDict:
    def __init__(self):
        self.dict = {}
    
    def get(self):
        return self.dict
    
    def keys(self):
        return list(self.dict.keys())
    
    def __getitem__(self, key):
        if key not in self.dict:
            self.dict[key] = []
        return self.dict[key]

def truncate_dict(dict):
    min_len = 1e9
    for k in dict.keys():
        min_len = min(min_len, len(dict[k]))
    
    for k in dict.keys():
        dict[k] = dict[k][:min_len]
    return dict


class DataReader:
    def __init__(self, COM_PORT, BAUD_RATES=921600, verbose=True, folder=None):
        self.ser = serial.Serial(COM_PORT, BAUD_RATES)
        self.cnt = 0
        self.skip = 0
        self.data_imu = ListDict()
        self.data_quat = ListDict()
        self.prev_sensor_name = None
        self.verbose = verbose
        
        self.folder = folder
        if folder:
            os.makedirs(folder, exist_ok=True)
        
        self.is_init = True
        self.start_time = time.time()
        self.end_time = time.time()
    
    def run(self):
        if not self.is_init:
            self.__init__()
        
        try:
            while True:
                while self.ser.in_waiting:
                    if self.cnt < self.skip:
                        self.cnt += 1
                        self.start_time = time.time()
                        continue
                    response = self.ser.readline().strip().decode()
                    self.parse_response(response)
                # uncomment this to stop recording in 60 seconds
                # if self.__len__() > 60:
                    # break
        except KeyboardInterrupt:
            self.is_init = False
            self.ser.close()
    
    def parse_response(self, response):
        if self.verbose:
            print(response)

        if response.startswith('S'):
            idx = response.find('#E')
            sensor_name = response[2]
            
            # sensor name history
            if self.prev_sensor_name is not None and sensor_name == self.prev_sensor_name:
                return
            self.prev_sensor_name = sensor_name
            
            # quaternion
            data_quat = response[2:idx].split('#')
            data_quat = [float(e) for e in data_quat[1:]]
            self.data_quat[sensor_name].append(data_quat)
            
            # imu
            # data_imu = response[idx+2:-2].split(', ')
            # data_imu = [float(e) for e in data_imu]
            # self.data_imu[sensor_name].append(data_imu)
        else:
            return
    
    def save(self, name=None):
        res = {
            'imu': truncate_dict(self.data_imu.get()),
            'quat': truncate_dict(self.data_quat.get()),
        }
        name = name or str(datetime.now()).split('.')[0].replace(':', '-')
        save_path = os.path.join(self.folder, f'{name}.pkl') if self.folder else f'{name}.pkl'
        pickle.dump(res, open(save_path, 'wb'))
        
        print('total time:', self.__len__())
        for k in self.data_quat.keys():
            print(k, len(self.data_quat[k]))
    
    def __len__(self):
        return time.time() - self.start_time

def serial_ports():
    ports = ['COM%s' % (i+1) for i in range(16)]
    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    print(result)
    exit()


def parse_args():
    parser = argparse.ArgumentParser(description="read imu data")
    #原來的
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--save", action='store_false')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    # serial_ports()
    # reader = DataReader(COM_PORT='COM6', verbose=args.verbose)
    reader = DataReader(COM_PORT='/dev/tty.usbserial-A50285BI', verbose=args.verbose)
    reader.run()
    if args.save:
        reader.save()
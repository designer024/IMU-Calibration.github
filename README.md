# IMU Calibration

## Setup
- 把模組組裝完成後插上電腦
- Run `serial_ports()` in `data/read_imu.py` to see currently used serial ports. 確定模組是接到COM幾。
- 把 DataReader 的 `COM_PORT` 改成模組在使用的。
- 跑 read_imu.py 並做 IMU 的校正。（每次斷電都需要重新校正）
- 校正完後就可以穿戴上模組，並開始蒐集資料。

## Dataset setup

### Data collecting
Run following command
```
cd data
python read_imu.py [--verbose] [--save]
```
- `[--verbose]`: print imu data at runtime or not. Default false.
- `[--save]`: save data or not. Data will save with file name `yyyy-mm-dd hh-mm-ss.pkl`. Default true.

### Generate dataset
Rearrange files as following folder structure.
```
data
├─ <dataset folder>
│　├─train
│　└─eval
└─ generate_dataset.py
```
Move recorded data to `train` and `eval` folders. Then run
```
python generate_dataset.py -f <dataset folder> -n <dataset name>
```


## Training
Run
```
python train.py --dataset_name <dataset name> 
```
Please refer to `train.py` for detail.

## Visualization

### IMU data 
The following command will show the visualization of the latest recorded data. For detail, please refer to the code.
```
python visualize_data.py
```

### 3D pose visualization
```
python gen_ske.py --file <path/to/imu/data>
```
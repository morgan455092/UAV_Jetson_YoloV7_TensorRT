# UAV_Jetson_YoloV7_TensorRT

使用TensorRT引擎於Jetson上運行YoloV7，執行無人機目標獲得及定位技術。

## 環境安裝

```
sudo apt-get update
```
```
sudo apt-get upgrade
```
```
sudo apt-get install -y liblapack-dev libblas-dev gfortran libfreetype6-dev libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev
```
```
sudo apt-get install -y python3-pip
```

Numpy 預裝了 Jetpack，因此請確保先卸載它，然後確認它是否已卸載。同時升級 pip3，然後安裝requirements.txt中的python套件：

```
pip3 install -r requirements.txt
```

如果PyYAML出現問題，請運行以下命令:

```
pip3 install “cython<3.0.0” & pip install --no-build-isolation pyyaml==6.0
```

安裝PyCuda，我們需要先導出幾個paths

```
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
```
```
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```
```
python3 -m pip install pycuda --user
```

安裝 Seaborn

```
sudo apt install python3-seaborn
```

安裝 torch & torchvision，若git clone那邊無法成功，可以直接到 https://github.com/pytorch/vision/releases/tag/v0.11.1 當中下載source code安裝。

```
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```
```
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```
```
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
```
```
cd torchvision
```
```
sudo python3 setup.py install 
```

以下不是必需的，但不錯的庫:

```
sudo python3 -m pip install -U jetson-stats==3.1.4
```

## 從.pt檔產生.wts檔，並從.wts生成.engine檔

Yolov7-tiny.pt 已在 repo 中提供。但是，如果您願意，您可以下載 yolov7 模型的任何其他版本。然後運行以下命令將 .pt 檔案轉換為 .wts 檔案

```
python3 gen_wts.py -w yolov7-tiny.pt -o yolov7-tiny.wts
```

在yolov7中創建一個build目錄。將生成的 wts 檔案複製並粘貼到 build 目錄中，然後運行以下命令。

**如果使用自定義模型，請確保在yolov7/include/config.h 中更新類別數量 kNumClas**

```
cd yolov7/
```
```
mkdir build
```
```
cd build
```
```
cp ../../yolov7-tiny.wts .
```
```
cmake ..
```
```
make 
```

生成.enigne

```
sudo ./yolov7 -s yolov7-tiny.wts  yolov7-tiny.engine t
```

測試.engine檔，這將對圖像進行推理，輸出將保存在 build 目錄中。

```
sudo ./yolov7 -d yolov7-tiny.engine ../images
```

## 參數配置

### 設定使用的engine file, 信心度、yolo版本

在app.py中進行修改

```python
# use path for library and engine file
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine="yolov7/build/yolov7-tiny.engine", conf=0.5, yolo_ver="v7")
```

### 使用CSI鏡頭或影片輸入

在app.py中取消註解。

```python
# 使用影片來源
# cap = cv2.VideoCapture("videos/testvideo.mp4")

# 使用CSI鏡頭
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
```

### 更新class數量

請確保在yolov7/include/config.h 中更新類別數量 kNumClas，再生成engine檔

```cpp
const static int kNumClass = 1;
```

請確保根據yolovDet.py中的類更新類別。

```python
    def __init__(self, library, engine, conf, yolo_ver):
        self.CONF_THRESH = conf 
        self.IOU_THRESHOLD = 0.4
        self.LEN_ALL_RESULT = 38001
        self.LEN_ONE_RESULT = 38
        self.yolo_version = yolo_ver
        self.categories = ["car"]
```

### 於命令列顯示fps, 類別、偵測框

取消app.py中註解

```python
    # for obj in detections:
    #    print(obj['class'], obj['conf'], obj['box'])
    # print("FPS: {} sec".format(1/t))
```

### 修改相機規格參數

修改yoloDet.py中的:

```python
    def get_target_position():
```


## 運行目標識別

app.py 用於對任何視頻檔或攝像機進行推理。

```
python3 app.py
```
import sys
import cv2 
import imutils
from yoloDet import YoloTRT
import twd97

def read_gps_data():
    with open('PLSATTIT_data.txt', 'r') as f:
        data = f.readlines()[-1]
        parts = data.split(',')
        latitude = parts[9]
        latitude = float(latitude)
        longitude = parts[10]
        longitude = float(longitude)
    return latitude, longitude

def PlotCord(img, latitude, longitude):
    text = '(' + str(longitude) + ', ' + str(latitude) + ')'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)  # 藍色
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness=1)
    height, width, channels = img.shape
    x = (width - text_width) // 2
    y = (height + text_height + baseline) // 2
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness=2)

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# read_gps_data('/dev/ttyUSB0', 115200)  # 根據實際情況修改端口名稱
#a = threading.Thread(target=read_gps_data, args=('/dev/ttyUSB1', 115200))
#a.start()

# use path for library and engine file
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine="yolov7/build/bestV2.engine", conf=0.5, yolo_ver="v7")

# 使用影片來源
# cap = cv2.VideoCapture("videos/testvideo.mp4")

# 使用CSI鏡頭
# cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

# USB
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1280)
    detections, t = model.Inference(frame)
    latitude_wgs84, longitude_wgs84 = read_gps_data()
    latitude_twd97, longitude_twd97 = twd97.fromwgs84(latitude_wgs84, longitude_wgs84)
    PlotCord(frame, latitude, longitude) 
    # for obj in detections:
    #    print(obj['class'], obj['conf'], obj['box'])
    # print("FPS: {} sec".format(1/t))
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
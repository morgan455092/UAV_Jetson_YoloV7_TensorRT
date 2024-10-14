import serial
from pymavlink import mavutil
import time

def parse_gnrmc(data):
    if data.startswith('$PLSATTIT'):
        parts = data.split(',')
        if parts[3] == '3':  # Check if the data is valid
            latitude = parts[9]
            latitude = float(latitude)
            longitude = parts[10]
            longitude = float(longitude)
            with open('PLSATTIT_data.txt', 'a') as file:
                file.write(data)
                file.write("\n")
            return latitude, longitude
    return None, None
# def parse_gnrmc(data):
#     if data.startswith('$GNRMC'):
#         parts = data.split(',')
#         if parts[2] == 'A':  # Check if the data is valid
#             latitude = convert_to_degrees(parts[3], parts[4])
#             longitude = convert_to_degrees(parts[5], parts[6])
#             return latitude, longitude
#     return None, None

def convert_to_degrees(value, direction):
    if len(value) == 11:  # 經度
        degrees = float(value[:3])
        minutes = float(value[3:])
    else:  # 緯度
        degrees = float(value[:2])
        minutes = float(value[2:])
    result = degrees + (minutes / 60)
    if direction in ['S', 'W']:
        result = -result
    return result

def read_gps_data(port, baudrate):
    master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=115200)
    master.wait_heartbeat()
    print("Heartbeat from system (system %u component %u)" % (master.target_system, master.target_component))
    
    with serial.Serial(port, baudrate, timeout=1) as ser:
        while True:
            line = ser.readline().decode('ascii', errors='replace').strip()
            latitude, longitude = parse_gnrmc(line)
            if latitude and longitude:
                lat = int(latitude * 1E7)
                lon = int(longitude * 1E7)

                # 輸出連續5次相同的數據，保持5Hz
                for _ in range(5):
                    print(f"Latitude: {lat}, Longitude: {lon}")
                    master.mav.gps_input_send(
                        0,  # Timestamp (micros since boot or Unix epoch)
                        0,  # ID of the GPS for multiple GPS inputs
                        (mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ |
                         mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_VEL_VERT |
                         mavutil.mavlink.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY),
                        0,  # GPS time (milliseconds from start of GPS week)
                        0,  # GPS week number
                        3,  # 0-1: no fix, 2: 2D fix, 3: 3D fix. 4: 3D with DGPS. 5: 3D with RTK
                        lat,  # Latitude (WGS84), in degrees * 1E7
                        lon,  # Longitude (WGS84), in degrees * 1E7
                        0,  # Altitude (AMSL, not WGS84), in m (positive for up)
                        0.7,  # GPS HDOP horizontal dilution of position in m
                        0.7,  # GPS VDOP vertical dilution of position in m
                        0,  # GPS velocity in m/s in NORTH direction in earth-fixed NED frame
                        0,  # GPS velocity in m/s in EAST direction in earth-fixed NED frame
                        0,  # GPS velocity in m/s in DOWN direction in earth-fixed NED frame
                        0,  # GPS speed accuracy in m/s
                        0,  # GPS horizontal accuracy in m
                        0,  # GPS vertical accuracy in m
                        7   # Number of satellites visible.
                    )
                    time.sleep(0.15)  # 每0.2秒輸出一次，達到5Hz

if __name__ == "__main__":
    read_gps_data('/dev/ttyUSB1', 115200)  # 根據實際情況修改端口名稱

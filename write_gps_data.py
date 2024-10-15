start_lat = 23.45055556
start_lon = 120.28611111
end_lat = 23.45222222
end_lon = 120.27861111

lat = start_lat
lon = start_lon

for _ in range(0,130):
    lat = lat + (end_lat - start_lat)/130
    lon = lon + (end_lon- start_lon)/130
    with open('PLSATTIT_data.txt', 'a') as f:
        f.write(str(lat) + "," + str(lon)+"\n")
    time.sleep(1)

lat = end_lat
lon = end_lon

for _ in range(0,130):
    lat = lat - (end_lat - start_lat)/130
    lon = lon - (end_lon- start_lon)/130
    with open('PLSATTIT_data.txt', 'a') as f:
        f.write(str(lat) + "," + str(lon)+"\n")
    time.sleep(1)
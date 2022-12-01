from tracemalloc import start
import geojson
from math import radians, cos, sin, asin, sqrt, atan
import numpy as np
import math
import networkx as nx
import queue
import copy
import folium
import json

station_path = "./data/bus_station.geojson"
line_path = "./data/bus_line.geojson"
poi_path = "./data/beijing_poi.csv"
json_path = "./data/line_station.json"
pos_path = "./data/pos_station.json"
subway_path = "./data/subway.csv"
duplicate_path = "./data/duplicate.csv"
taxi_path = "20220301.csv"


def overlap_rate(line, total_lines, lk):
    nums = 0
    for key in total_lines:
        if key != lk:
            line_tmp = total_lines[key]
            for i in range(len(line_tmp)):
                if line_tmp[i] in line:
                    nums += 1
                    break
    return float(nums/len(total_lines.keys()))


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(
        radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2*asin(sqrt(a))*6371
    distance = round(distance, 3)
    return distance


# information of all the stations
with open(station_path, 'r') as f:
    gj = geojson.load(f)
data = gj["features"]
stations = []
reverse_stations = []
for i in range(len(data)):
    pos = data[i]["geometry"]["coordinates"]
    stations.append(pos)
    reverse_stations.append(pos[::-1])

# poi points in Tongzhou district
tong_points = []
with open(poi_path, "r") as f:
    poi_data = f.readlines()
    for line in poi_data:
        line_tmp = line.strip().split(",")
        if line_tmp[-1] == "通州区":
            try:
                pos = [float(line_tmp[3]), float(line_tmp[4])]
                tong_points.append(pos)
            except:
                pass


with open(line_path, 'r') as f:
    gj = geojson.load(f)
data = gj["features"]
line_new = []

# 583路(翠福园小区--地铁黄渠站)
# 583路(地铁黄渠站--翠福园小区)
# 582路(怡乐南街公交场站--地铁通州北关站)
# 582路(地铁通州北关站--怡乐南街公交场站)
# 372路(怡乐南街公交场站--日新路南口)
# 372路(日新路南口--怡乐南街公交场站)
key = "372路(日新路南口--怡乐南街公交场站)"
print(key)
for i in range(len(data)):
    if data[i]["properties"]["NAME"] == key:
        line = data[i]["geometry"]["coordinates"][0]

# line_info: coordinates of each station in corresponding line (key)
with open(json_path, "r", encoding="utf-8") as f:
    line_info = json.load(f)
actual_stations = line_info[key]
print("total lines:", len(line_info.keys()))

# pos_info: key: name of bus station, val: corresponding coordinate (list)
# info_pos: key: coordinate, val: name of bus station
with open(pos_path, "r", encoding="utf-8") as f:
    pos_info = json.load(f)
info_pos = dict()
for key in pos_info:
    for val in pos_info[key]:
        info_pos[tuple(val)] = key
print(list(info_pos.keys())[:5])

bus_subway = dict()
with open(subway_path, "r", encoding="utf-8") as f:
    poi_data = f.readlines()
    for i in range(1, len(poi_data)):
        infos = poi_data[i].split(',')[4]
        sub_info = poi_data[i].split(',')[1]
        if infos not in bus_subway:
            bus_subway[infos] = [sub_info]
        else:
            bus_subway[infos].append(sub_info)

# key: coordinate of bus station, val: number of subway station
bus_subway_new = dict()
for key in bus_subway:
    for val in pos_info[key]:
        bus_subway_new[tuple(val)] = len(set(bus_subway[key]))

# key: segment info, val: occur_num
duplicate_dict = dict()
with open(duplicate_path, 'r', encoding='utf-8') as f:
    infos1 = f.readlines()
    for i in range(1, len(infos1)):
        line_start = infos1[i].split(',')[5]
        line_end = infos1[i].split(',')[6]
        occur_num = int(infos1[i].split(',')[7])
        duplicate_dict[tuple([line_start, line_end])] = occur_num
print(list(duplicate_dict.keys())[:5])

# flow dict: key is lng-lat pair and value is flow number
flow_num_dict = dict()
with open(taxi_path, "r") as f:
    taxi_data = f.readlines()
    for i in range(1, len(taxi_data)):
        infos = taxi_data[i].strip().split(',')
        lng_info = float(infos[1])
        lat_info = float(infos[2])
        flow_info = int(infos[4])
        if flow_info == 1 and tuple([lng_info, lat_info]) not in flow_num_dict:
            flow_num_dict[tuple([lng_info, lat_info])] = 1
        elif flow_info == 1:
            flow_num_dict[tuple([lng_info, lat_info])] += 1
# =====================================================================================================

line_lng = []
line_lat = []
for i in range(len(line)):
    line_lng.append(line[i][0])
    line_lat.append(line[i][1])
    line_new.append(line[i][::-1])
print(max(line_lat))
print(min(line_lat))
start_point = line[0]
end_point = line[-1]
start_lng = min(start_point[0], end_point[0])
start_lat = min(start_point[1], end_point[1])
end_lng = max(end_point[0], start_point[0])
end_lat = max(end_point[1], start_point[1])
print(start_lng, start_lat, end_lng, end_lat)
candidate_stops = []

expand_range = 0.2
for item in stations:
    if item[0] > start_lng-expand_range and item[0] < end_lng+expand_range and item[1] < end_lat+expand_range and item[1] > start_lat-expand_range:
        candidate_stops.append(item)
candidate_pois = []
for item in tong_points:
    if item[0] > start_lng-expand_range and item[0] < end_lng+expand_range and item[1] < end_lat+expand_range and item[1] > start_lat-expand_range:
        candidate_pois.append(item)
candidiate_flow_dict = dict()
for key in flow_num_dict:
    if key[0] > start_lng-expand_range and key[0] < end_lng+expand_range and key[1] < end_lat+expand_range and key[1] > start_lat-expand_range:
        candidiate_flow_dict[key] = flow_num_dict[key]

G = nx.Graph()
nodes_tmp = list(range(len(candidate_stops)+2))
nodes = list(map(str, nodes_tmp))
edges = []

# use queue to build graph from start point to end point
q = queue.Queue()
q.put(start_point)
tags = [True]*len(candidate_stops)
times = 0
start_dis = 0.25
end_dis = 0.75
while not q.empty():
    ss = q.qsize()
    for i in range(ss):
        if times == 0:
            temp = q.get()
            for j in range(len(candidate_stops)):
                if geodistance(temp[0], temp[1], candidate_stops[j][0], candidate_stops[j][1]) <= end_dis and geodistance(temp[0], temp[1], candidate_stops[j][0], candidate_stops[j][1]) >= start_dis:
                    edges.append((str(0), str(j+1)))
                    q.put(j)
                    tags[j] = False
        else:
            inx = q.get()
            for j in range(len(candidate_stops)):
                if geodistance(candidate_stops[inx][0], candidate_stops[inx][1], candidate_stops[j][0], candidate_stops[j][1]) <= end_dis and geodistance(candidate_stops[inx][0], candidate_stops[inx][1], candidate_stops[j][0], candidate_stops[j][1]) >= start_dis:
                    edges.append((str(inx + 1), str(j + 1)))
                    if tags[j]:
                        q.put(j)
                        tags[j] = False
    times += 1

nums = 0
for j in range(len(candidate_stops)):
    if geodistance(end_point[0], end_point[1], candidate_stops[j][0], candidate_stops[j][1]) <= end_dis and geodistance(end_point[0], end_point[1], candidate_stops[j][0], candidate_stops[j][1]) >= start_dis:
        edges.append((str(j + 1), str(len(candidate_stops) + 1)))
        nums += 1

print("neighbors of end point:", nums)
G.add_nodes_from(nodes)
G.add_edges_from(edges)
print("edges of graph:", len(G.edges))


def trans(index):
    index = int(index)
    if index == 0:
        return start_point
    elif index == len(candidate_stops)+1:
        return end_point
    return candidate_stops[int(index)-1]


def get_flow(point):
    total_flows = 0
    for key in candidiate_flow_dict:
        if geodistance(key[0], key[1], point[0], point[1]) <= 0.15:
            total_flows += candidiate_flow_dict[key]
    return total_flows


def get_score(line):
    total_score = 0
    total_pois = 0
    total_subways = 0
    total_duplicates = 0
    total_flows = 0
    for i in range(len(line)):
        num_poi = 0
        for poi in candidate_pois:
            if geodistance(poi[0], poi[1], line[i][0], line[i][1]) <= 0.1:  # less than 100m
                num_poi += 1
        total_pois += num_poi
        num_subway = 0
        if tuple(line[i]) in bus_subway_new:
            num_subway = bus_subway_new[tuple(line[i])]
        total_subways += num_subway
        duplicate_num = 0
        if i < len(line)-1:
            if tuple([info_pos[tuple(line[i])], info_pos[tuple(line[i+1])]]) in duplicate_dict:
                duplicate_num += duplicate_dict[tuple(
                    [info_pos[tuple(line[i])], info_pos[tuple(line[i+1])]])]
            elif tuple([info_pos[tuple(line[i+1])], info_pos[tuple(line[i])]]) in duplicate_dict:
                duplicate_num += duplicate_dict[tuple(
                    [info_pos[tuple(line[i+1])], info_pos[tuple(line[i])]])]
            else:
                duplicate_num += 1
        total_duplicates += duplicate_num
        flow_num = get_flow(line[i])
        total_flows += flow_num
        total_score = total_score+0.4*num_poi+0.6 * \
            num_subway-0.25*duplicate_num+flow_num
    print("total pois:", total_pois)
    print("total subways:", total_subways)
    print("total duplicates:", total_duplicates)
    print("total flow:", total_flows)
    return float(total_score/len(line))


print("length of actual line:", len(actual_stations))
print("avg score of actual line:", get_score(actual_stations))


def judge_line(start_node, end_node, G, ends):
    judge = False
    start_lng, start_lat = start_node
    end_lng, end_lat = end_node
    theta = atan(abs(end_lat - start_lat) / abs(end_lng - start_lng))
    epsi = 1

    def criter1(line):
        line = list(map(lambda x: trans(x), line))
        for i in range(1, len(line)):
            if geodistance(line[i][0], line[i][1], line[i-1][0], line[i-1][1]) > epsi:
                return False
        return True

    def criter2(line):
        line = list(map(lambda x: trans(x), line))
        line = np.array(line)
        line_new = line[1:line.shape[0], 0] * \
            math.cos(theta)+line[1:line.shape[0], 1]*math.sin(theta)
        return all([line_new[i] < line_new[i+1] for i in range(line_new.shape[0]-1)])

    def criter3(line):
        line = list(map(lambda x: trans(x), line))
        dis_diff = list(map(lambda x: geodistance(
            start_lng, start_lat, x[0], x[1]), line[1:len(line)]))
        return all([dis_diff[i] < dis_diff[i+1] for i in range(len(dis_diff)-1)])

    def criter4(line):
        line = list(map(lambda x: trans(x), line))
        dis_diff = list(map(lambda x: geodistance(
            end_lng, end_lat, x[0], x[1]), line[1:len(line)]))
        return all([dis_diff[i] > dis_diff[i+1] for i in range(len(dis_diff)-1)])

    def criter5(line):
        line = list(map(lambda x: trans(x), line))
        for i in range(2, len(line)):
            lng_tmp = line[i][0]
            lat_tmp = line[i][1]
            dis_diff = np.array(
                list(map(lambda x: geodistance(lng_tmp, lat_tmp, x[0], x[1]), line[:i])))
            if np.argmin(dis_diff) != dis_diff.shape[0]-1:
                return False
        return True

    def cal_score(start_point, point):
        pos_point = trans(point)
        num_poi = 0
        for poi in candidate_pois:
            if geodistance(poi[0], poi[1], pos_point[0], pos_point[1]) <= 0.1:
                num_poi += 1
        num_subway = 0
        if tuple(pos_point) in bus_subway_new:
            num_subway = bus_subway_new[tuple(pos_point)]
        num_duplicate = 1
        if tuple([info_pos[tuple(trans(start_point))], info_pos[tuple(pos_point)]]) in duplicate_dict:
            num_duplicate = duplicate_dict[tuple(
                [info_pos[tuple(trans(start_point))], info_pos[tuple(pos_point)]])]
        elif tuple([info_pos[tuple(pos_point)], info_pos[tuple(trans(start_point))]]) in duplicate_dict:
            num_duplicate = duplicate_dict[tuple(
                [info_pos[tuple(pos_point)], info_pos[tuple(trans(start_point))]])]
        num_flow = get_flow(pos_point)
        total_score = 0.4*num_poi+0.6*num_subway-0.25*num_duplicate+num_flow
        return total_score

    def explore_dfs(starts, ends, line):
        global judge
        if judge:
            return

        if starts == ends:
            judge = True
            return

        for node in G.neighbors(starts):
            if len(line) >= 3 and criter3(line) and criter4(line) and criter5(line):
                line.append(node)
                explore_dfs(node, ends, line)
                line.pop()
            elif len(line) < 3:
                line.append(node)
                explore_dfs(node, ends, line)
                line.pop()

    lines = dict()
    topk = 25

    def explore_pois(starts, ends, line):
        global judge
        if len(line) == 1:
            topk = 25
        elif len(line) == 2:
            topk = 2
        else:
            topk = 1
        if starts == ends:
            print("suitable line:", line)
            ts = 0
            for i in range(1, len(line)):
                ts += cal_score(line[i-1], line[i])
            lines[tuple(line)] = float(ts/len(line))
            return
        poi_dict = dict()
        for node in G.neighbors(starts):
            judge = False
            line.append(node)
            explore_dfs(node, ends, line)
            if judge:
                poi_dict[node] = cal_score(starts, node)
            line.pop()

        if len(poi_dict) == 0:
            return
        poi_tmp = sorted(poi_dict.items(), key=lambda x: x[1], reverse=True)
        cans_node = []
        if len(poi_tmp) < topk:
            for i in range(len(poi_tmp)):
                cans_node.append(poi_tmp[i][0])
        else:
            for i in range(topk):
                cans_node.append(poi_tmp[i][0])
        for node in G.neighbors(starts):
            if node in cans_node:
                line.append(node)
                explore_pois(node, ends, line)
                line.pop()

    node_start_tmp = str(0)
    node_end_tmp = str(ends)
    explore_pois(node_start_tmp, node_end_tmp, [node_start_tmp])
    return lines


lines = judge_line(start_point, end_point, G, len(candidate_stops)+1)
a = sorted(lines.items(), key=lambda x: x[1], reverse=True)

temp_line = list(a[0][0])
tml = []
get_line = [start_point[::-1]]  # get_line is for plotting line in folium
for i in range(1, len(temp_line)-1):
    get_line.append(candidate_stops[int(temp_line[i])-1][::-1])
get_line.append(end_point[::-1])
for i in range(len(temp_line)):
    tml.append(trans(temp_line[i]))
print("length of getting line:", len(tml))
print("avg score of getting line:", get_score(tml))

m = folium.Map(line_new[0], zoom_start=14)
route = folium.PolyLine(
    line_new, weight=3, color="blue", opacity=0.8).add_to(m)
route1 = folium.PolyLine(
    get_line, weight=3, color="red", opacity=0.8).add_to(m)
# folium.Marker(line_new[0], popup='<b>Starting Point</b>').add_to(m)
# for i in range(1,len(get_line)-1):
#     folium.Marker(get_line[i], popup=f'<b>{i} Point</b>').add_to(m)
# folium.Marker(line_new[-1], popup='<b>Ending Point</b>').add_to(m)
m.save("bplanner_372_2.html")

from tracemalloc import start
import geojson
from math import radians, cos, sin, asin, sqrt, atan
import numpy as np
import math
import networkx as nx
import queue
import json
import os
import math

station_path = "./data/bus_station.geojson"
line_path = "./data/bus_line.geojson"
poi_path = "./data/beijing_poi.csv"
json_path = "./data/line_station.json"
pos_path = "./data/pos_station.json"
subway_path = "./data/subway.csv"
duplicate_path = "./data/duplicate.csv"
graph_folder = "./data/graphs_test"
feature_folder = "./data/features_test"

if not os.path.exists(graph_folder):
    os.mkdir(graph_folder)
if not os.path.exists(feature_folder):
    os.mkdir(feature_folder)


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(
        radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2*asin(sqrt(a))*6371
    distance = round(distance, 3)
    return distance


# point1: start point2: end 
def judges(point1, point2, point3):
    temp1 = [point2[0]-point1[0], point2[1]-point1[1]]
    temp2 = [point3[0]-point1[0], point3[1]-point1[1]]
    if temp1[0]*temp2[0]+temp2[1]*temp1[1]>0:
        return True
    return False

def dist(point1, point2):
    xx = (point2[0]-point1[0])**2
    yy = (point2[1]-point1[1])**2
    return math.sqrt(xx+yy)

def cal_dist(point1, point2, point3):
    line1 = [point2[0]-point1[0], point2[1]-point1[1]]
    dist1 = dist(point1, point2)
    line2 = [point3[0]-point1[0], point3[1]-point1[1]]
    dist2 = dist(point1, point3)
    temp = line1[0]*line2[0]+line1[1]*line2[1]
    cos_ang = temp/(dist1*dist2)
    gd = geodistance(point3[0], point3[1], point1[0], point1[1])
    return gd*cos_ang

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
        try:
            pos = [float(line_tmp[3]), float(line_tmp[4])]
            tong_points.append(pos)
        except:
            pass


with open(line_path, 'r') as f:
    gj = geojson.load(f)
data_new = gj["features"]

with open(json_path, "r", encoding="utf-8") as f:
    line_info = json.load(f)

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

bus_subway_new = dict()
for key in bus_subway:
    for val in pos_info[key]:
        bus_subway_new[tuple(val)] = len(set(bus_subway[key]))

duplicate_dict = dict()
with open(duplicate_path, 'r', encoding='utf-8') as f:
    infos1 = f.readlines()
    for i in range(1, len(infos1)):
        line_start = infos1[i].split(',')[5]
        line_end = infos1[i].split(',')[6]
        occur_num = int(infos1[i].split(',')[7])
        duplicate_dict[tuple([line_start, line_end])] = occur_num
print(list(duplicate_dict.keys())[:5])

def generate(line, id, name):
    line_lng = []
    line_lat = []
    for i in range(len(line)):
        line_lng.append(line[i][0])
        line_lat.append(line[i][1])
    print(max(line_lat))
    print(min(line_lat))
    start_point = line[0]
    end_point = line[-1]
    min_lng = min(start_point[0], end_point[0])
    min_lat = min(start_point[1], end_point[1])
    max_lng = max(end_point[0], start_point[0])
    max_lat = max(end_point[1], start_point[1])
    print(min_lng, min_lat, max_lng, max_lat)
    candidate_stops = []
    
    expand_range = 0.15
    for item in stations:
        if judges(start_point, end_point, item) and judges(end_point, start_point, item) and item[0] >= min_lng-expand_range and item[0] <= max_lng+expand_range and item[1] <= max_lat+expand_range and item[1] >= min_lat-expand_range:
            candidate_stops.append(item)
    candidate_pois = []
    expand_range1 = 0.151
    for item in tong_points:
        if item[0] >= min_lng-expand_range1 and item[0] <= max_lng+expand_range1 and item[1] <= max_lat+expand_range1 and item[1] >= min_lat-expand_range1:
            candidate_pois.append(item)
    
    def trans(index):
        index = int(index)
        if index == 0:
            return start_point
        elif index == len(candidate_stops)+1:
            return end_point
        return candidate_stops[int(index)-1]
    
    def get_edge_feature(start_point, point):
        pos_point = trans(point)
        num_duplicate = 1
        if tuple([info_pos[tuple(trans(start_point))], info_pos[tuple(pos_point)]]) in duplicate_dict:
            num_duplicate = duplicate_dict[tuple(
                [info_pos[tuple(trans(start_point))], info_pos[tuple(pos_point)]])]
        elif tuple([info_pos[tuple(pos_point)], info_pos[tuple(trans(start_point))]]) in duplicate_dict:
            num_duplicate = duplicate_dict[tuple(
                [info_pos[tuple(pos_point)], info_pos[tuple(trans(start_point))]])]
        return float(1/num_duplicate)
    
    G = nx.Graph()
    nodes_tmp = list(range(len(candidate_stops)+2))
    nodes = list(map(str, nodes_tmp))
    edges = []
    
    q = queue.Queue()
    q.put(start_point)
    tags = [True]*len(candidate_stops)
    times = 0
    start_dis = 0.3
    end_dis = 0.65
    node_set = []
    while not q.empty():
        ss = q.qsize()
        for i in range(ss):
            if times == 0:
                temp = q.get()
                for j in range(len(candidate_stops)):
                    if candidate_stops[j][0]==end_point[0] and candidate_stops[j][1]==end_point[1]:
                        continue
                    dist_tmp = geodistance(temp[0], temp[1], candidate_stops[j][0], candidate_stops[j][1])
                    if dist_tmp <= end_dis and dist_tmp >= start_dis:
                        edge_weight = get_edge_feature(str(0), str(j+1))
                        edges.append((str(0), str(j+1), str(edge_weight)))
                        q.put(j)
                        node_set.append(j)
                        tags[j] = False
            else:
                inx = q.get()
                for j in range(len(candidate_stops)):
                    if candidate_stops[j][0]==end_point[0] and candidate_stops[j][1]==end_point[1]:
                        continue
                    dist_tmp = geodistance(candidate_stops[inx][0], candidate_stops[inx][1], candidate_stops[j][0], candidate_stops[j][1])
                    if dist_tmp <= end_dis and dist_tmp >= start_dis:
                        edge_weight = get_edge_feature(str(inx+1), str(j+1))
                        edges.append((str(inx + 1), str(j + 1), str(edge_weight)))
                        if tags[j]:
                            q.put(j)
                            node_set.append(j)
                            tags[j] = False
        times += 1
    
    nums = 0
    for j in node_set:
        val_tmp = geodistance(end_point[0], end_point[1], candidate_stops[j][0], candidate_stops[j][1])
        if val_tmp <= end_dis and val_tmp >= start_dis:
            edge_weight = get_edge_feature(str(j+1), str(len(candidate_stops) + 1))
            edges.append((str(j + 1), str(len(candidate_stops) + 1), str(edge_weight)))
            nums += 1
    print("neighbors of end point:", nums)
    if nums==0:
        return False
    
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    # nodes2remove = [node for node in G.nodes() if G.degree(node)==0]
    # for node_tmp in nodes2remove:
    #     G.remove_node(node_tmp)
    print("nodes of graph:", len(G.nodes))
    print("edges of graph:", len(G.edges))
    
    
    def cal_score(point):
        pos_point = trans(point)
        num_poi = 0
        for poi in candidate_pois:
            if geodistance(poi[0], poi[1], pos_point[0], pos_point[1]) <= 0.1:
                num_poi += 1
        num_subway = 0
        if tuple(pos_point) in bus_subway_new:
            num_subway = bus_subway_new[tuple(pos_point)]
        num_duplicate = 0
        for start_point in G.neighbors(point):
            if tuple([info_pos[tuple(trans(start_point))], info_pos[tuple(pos_point)]]) in duplicate_dict:
                num_duplicate_tmp = duplicate_dict[tuple(
                    [info_pos[tuple(trans(start_point))], info_pos[tuple(pos_point)]])]
            elif tuple([info_pos[tuple(pos_point)], info_pos[tuple(trans(start_point))]]) in duplicate_dict:
                num_duplicate_tmp = duplicate_dict[tuple(
                    [info_pos[tuple(pos_point)], info_pos[tuple(trans(start_point))]])]
            else:
                num_duplicate_tmp = 0
            num_duplicate+=num_duplicate_tmp
        if G.degree(point)>0:
            avg_dumplicate = float(num_duplicate/G.degree(point))
        else:
            avg_dumplicate = 0
        total_score = 0.4*num_poi+0.6*num_subway-0.25*avg_dumplicate
        return total_score, num_poi, num_subway, avg_dumplicate
    
    score_dict = dict()
    for nn in G.nodes():
        _, pois, subways, avg_dumplics = cal_score(nn)
        st = trans(0)
        cur_pos = trans(nn)
        en = trans(len(candidate_stops)+1)
        st_geodist = geodistance(start_point[0], start_point[1], cur_pos[0], cur_pos[1])
        en_geodist = geodistance(end_point[0], end_point[1], cur_pos[0], cur_pos[1])
        if st_geodist==0:
            st_dist = 0
            en_dist = en_geodist
        elif en_geodist==0:
            en_dist = 0
            st_dist = st_geodist
        else:
            st_dist = cal_dist(st, en, cur_pos)
            en_dist = cal_dist(en, st, cur_pos)
        score_dict[nn] = [pois, subways, avg_dumplics, st_dist, en_dist, st_geodist, en_geodist]

    node_infos = []
    for nd in G.nodes():
        pois, subways, avg_dumplics, st_dist, en_dist, st_geodist, en_geodist = score_dict[nd]
        node_infos.append([nd, str(pois), str(subways), str(avg_dumplics), str(st_dist), str(en_dist), str(st_geodist), str(en_geodist)])
    
    edge_infos = G.edges(data=True)
    save_graph_path = os.path.join(graph_folder, str(id)+".txt")
    with open(save_graph_path, "w") as f:
        f.write(name+"\n")
        for ee in edge_infos:
            temp = [str(ee[0]), str(ee[1]), str(ee[2]["weight"])]
            temp1 = ",".join(temp)+"\n"
            f.write(temp1)
    save_feature_path = os.path.join(feature_folder, str(id)+".txt")
    with open(save_feature_path,"w") as f:
        f.write(name+"\n")
        for ii in node_infos:
            temp1 = ",".join(ii)+"\n"
            f.write(temp1)
    return True
        

if __name__ == "__main__":
    nums = 20
    times = 0
    for i in range(len(data_new)):
        print("key:", data_new[i]["properties"]["NAME"])
        name = data_new[i]["properties"]["NAME"]
        line = data_new[i]["geometry"]["coordinates"][0]
        res = generate(line, i, name)
        if res:
            times+=1
        if times>=nums:
            break
        
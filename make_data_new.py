from tracemalloc import start
import geojson
from math import radians, cos, sin, asin, sqrt, atan
import numpy as np
import networkx as nx
import queue
import copy
import json
import os
import random
import time
import math

station_path = "./data/bus_station.geojson"
line_path = "./data/bus_line.geojson"
poi_path = "./data/beijing_poi.csv"
json_path = "./data/line_station.json"
pos_path = "./data/pos_station.json"
subway_path = "./data/subway.csv"
duplicate_path = "./data/duplicate.csv"
graph_folder = "./data/graphs_new"
feature_folder = "./data/features_new"

seed = 8
random.seed(seed)
np.random.seed(seed)

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

# point1: start point2: end 
def judges(point1, point2, point3):
    temp1 = [point2[0]-point1[0], point2[1]-point1[1]]
    temp2 = [point3[0]-point1[0], point3[1]-point1[1]]
    if temp1[0]*temp2[0]+temp2[1]*temp1[1]>=0:
        return True
    return False

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

def generate(point):
    judge = False
    lng = point[0]
    lat = point[1]
    choice = np.random.choice([1,2,3,4])
    dist = []
    start_lng = lng
    start_lat = lat
    start_point = [start_lng, start_lat]
    degree = 0.025
    if choice == 1:
        end_lng = lng+degree
        end_lat = lat+degree
    elif choice == 2:
        end_lng = lng+degree
        end_lat = lat-degree
    elif choice == 3:
        end_lng = lng-degree
        end_lat = lat-degree
    else:
        end_lng = lng-degree
        end_lat = lat+degree
    end_point = [end_lng, end_lat]
    indexes = []
    for j in range(len(stations)):
        if judges(start_point, end_point, stations[j]) and judges(end_point, start_point, stations[j]): 
            dist.append(geodistance(end_lng, end_lat, stations[j][0], stations[j][1]))
            indexes.append(j)
    min_index = np.argsort(np.array(dist))[0]
    end_point = stations[indexes[min_index]]
    
    min_lng = min(start_point[0], end_point[0])
    max_lng = max(start_point[0], end_point[0])
    min_lat = min(start_point[1], end_point[1])
    max_lat = max(start_point[1], end_point[1])
    G = nx.Graph()
        
    candidate_stops = []
    expand_range1 = 0.01
    for item in stations:
        if judges(start_point, end_point, item) and judges(end_point, start_point, item) and item[0] >= min_lng-expand_range1 and item[0] <= max_lng+expand_range1 and item[1] <= max_lat+expand_range1 and item[1] >= min_lat-expand_range1:
            candidate_stops.append(item)
    if len(candidate_stops)<50:
        return dict(), dict(), G
    
    candidate_pois = []
    expand_range = 0.011
    for item in tong_points:
        if item[0] >= min_lng-expand_range and item[0] <= max_lng+expand_range and item[1] <= max_lat+expand_range and item[1] >= min_lat-expand_range:
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
    distss = []
    for j in node_set:
        distss.append(geodistance(end_point[0], end_point[1], candidate_stops[j][0], candidate_stops[j][1]))
        if geodistance(end_point[0], end_point[1], candidate_stops[j][0], candidate_stops[j][1]) <= end_dis and geodistance(end_point[0], end_point[1], candidate_stops[j][0], candidate_stops[j][1]) >= start_dis:
            edge_weight = get_edge_feature(str(j+1), str(len(candidate_stops) + 1))
            edges.append((str(j + 1), str(len(candidate_stops) + 1), str(edge_weight)))
            nums += 1
    distss = sorted(distss)
    print(distss[:3])
    print("neighbors of end point:", nums)
    if nums==0:
        return dict(), dict(), G
    
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    print("nodes of graph:", len(G.nodes))
    print("edges of graph:", len(G.edges))
    if len(G.nodes)>275:
        return dict(), dict(), G
    
    def get_features(point):
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
        score, pois, subways, avg_dumplics = get_features(nn)
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
        score_dict[nn] = [score, pois, subways, avg_dumplics, st_dist, en_dist, st_geodist, en_geodist]
        
    def criter1(line):
        line = list(map(lambda x: trans(x), line))
        dis_diff = list(map(lambda x: geodistance(
            start_lng, start_lat, x[0], x[1]), line[1:len(line)]))
        return all([dis_diff[i] < dis_diff[i+1] for i in range(len(dis_diff)-1)])

    def criter2(line):
        line = list(map(lambda x: trans(x), line))
        dis_diff = list(map(lambda x: geodistance(
            end_lng, end_lat, x[0], x[1]), line[1:len(line)]))
        return all([dis_diff[i] > dis_diff[i+1] for i in range(len(dis_diff)-1)])

    def criter3(line):
        line = list(map(lambda x: trans(x), line))
        for i in range(2, len(line)):
            lng_tmp = line[i][0]
            lat_tmp = line[i][1]
            dis_diff = np.array(
                list(map(lambda x: geodistance(lng_tmp, lat_tmp, x[0], x[1]), line[:i])))
            if np.argmin(dis_diff) != dis_diff.shape[0]-1:
                return False
        return True
    
    def explore_dfs(starts, ends, line):
        global judge
        if judge:
            return
        if starts == ends:
            judge = True
            return

        for node in G.neighbors(starts):
            if (len(line) >= 3 and criter1(line) and criter2(line) and criter3(line)) or (len(line) < 3 and criter1(line) and criter2(line)):
                line.append(node)
                explore_dfs(node, ends, line)
                line.pop()
    
    lines = dict()
    start_time = time.time()
    def explore_pois(starts, ends, line):
        global judge
        end_time = time.time()
        if(end_time-start_time)/3600 > 2:
            return
        if len(line)==1:
            topk = G.degree(line[0])
        elif len(line) == 2:
            topk = 2
        else:
            topk = 1
        if starts == ends:
            print("suitable line:", line)
            ts = 0
            for i in range(len(line)):
                ts += score_dict[line[i]][0]
            lines[tuple(line)] = float(ts/len(line))
            return
        poi_dict = dict()
        for node in G.neighbors(starts):
            judge = False
            line.append(node)
            explore_dfs(node, ends, line)
            if judge:
                poi_dict[node] = score_dict[node][0]
            line.pop()

        if len(poi_dict) == 0:
            return
        poi_tmp = sorted(poi_dict.items(), key=lambda x: x[1], reverse=True)
        cans_node = []
        for i in range(min(topk, len(poi_tmp))):
            cans_node.append(poi_tmp[i][0])
        for node in G.neighbors(starts):
            if node in cans_node:
                line.append(node)
                explore_pois(node, ends, line)
                line.pop()

    node_start_tmp = str(0)
    node_end_tmp = str(len(candidate_stops)+1)
    explore_pois(node_start_tmp, node_end_tmp, [node_start_tmp])
    return lines, score_dict, G

if __name__ == '__main__':
    times = 0
    for i in range(len(data_new)):
        print("key:", data_new[i]["properties"]["NAME"])
        name = data_new[i]["properties"]["NAME"]
        # line = data_new[i]["geometry"]["coordinates"][0]
        line = line_info[name]
        random.shuffle(line)
        for j in range(len(line)):
            lines, score_dict, graph = generate(line[j])
            if len(lines)==0:
                continue
            line_best_index = sorted(lines.items(), key=lambda x: x[1], reverse=True)
            print("number of lines:", len(line_best_index))
            line_best = list(line_best_index[0][0])
            print("best line:", line_best)
            
            points1 = []
            jm = 3
            if len(line_best_index)<jm:
                for k in range(len(line_best_index)):
                    line_tmp = list(line_best_index[k][0])
                    points1.extend(line_tmp)
            else:
                nums = max(jm, int(0.4*len(line_best_index)))
                for k in range(nums):
                    line_tmp = list(line_best_index[k][0])
                    points1.extend(line_tmp)
            points1 = list(set(points1))
            print("ratio:", float(len(points1)/len(graph.nodes)))
            
            save_feature_path = os.path.join(feature_folder, str(i)+".txt")
            with open(save_feature_path,"w") as f:
                f.write(name+"\n")
                for node in graph.nodes:
                    if node in points1:
                        category = str(1)
                    else:
                        category = str(0)
                    _, num_poi, num_subway, avg_dumplicate, st_dist, en_dist, st_geodist, en_geodist = score_dict[node]
                    f_tmp = [node, category, str(num_poi), str(num_subway), str(avg_dumplicate), str(st_dist), str(en_dist), str(st_geodist), str(en_geodist)]
                    temp1 = ",".join(f_tmp)+"\n"
                    f.write(temp1)
            
            edge_infos = graph.edges(data=True)
            save_graph_path = os.path.join(graph_folder, str(i)+".txt")
            with open(save_graph_path, "w") as f:
                f.write(name+"\n")
                for ee in edge_infos:
                    temp = [str(ee[0]), str(ee[1]), str(ee[2]["weight"])]
                    temp1 = ",".join(temp)+"\n"
                    f.write(temp1)
            times+=1
            break
                


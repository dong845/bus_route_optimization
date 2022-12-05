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
import random
import time

station_path = "./data/bus_station.geojson"
line_path = "./data/bus_line.geojson"
poi_path = "./data/beijing_poi.csv"
json_path = "./data/line_station.json"
pos_path = "./data/pos_station.json"
subway_path = "./data/subway.csv"
duplicate_path = "./data/duplicate.csv"

seed = 8
random.seed(seed)
np.random.seed(seed)

def judges(point1, point2, point3):
    temp1 = [point2[0]-point1[0], point2[1]-point1[1]]
    temp2 = [point3[0]-point1[0], point3[1]-point1[1]]
    if temp1[0]*temp2[0]+temp2[1]*temp1[1]>=0:
        return True
    return False


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

def cal_cos(point1, point2, point3):
    if point3==point2 or point3==point1:
        return 0
    line1 = [point2[0]-point1[0], point2[1]-point1[1]]
    dist1 = dist(point1, point2)
    line2 = [point3[0]-point1[0], point3[1]-point1[1]]
    dist2 = dist(point1, point3)
    temp = line1[0]*line2[0]+line1[1]*line2[1]
    cos_ang = temp/(dist1*dist2)
    return cos_ang

def plot_line(line1, line2):
    # print("line1:", line1)
    print("line2:", line2)
    m = folium.Map(line1[0], zoom_start=14)
    route = folium.PolyLine(
        line1, weight=3, color="blue", opacity=0.8).add_to(m)
    route1 = folium.PolyLine(
        line2, weight=3, color="red", opacity=0.8).add_to(m)
    m.save("test.html")


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
key = "583路(翠福园小区--地铁黄渠站)"
print(key)
for i in range(len(data)):
    if data[i]["properties"]["NAME"] == key:
        line = data[i]["geometry"]["coordinates"][0]

# line_info: coordinates of each station in corresponding line (key)
with open(json_path, "r", encoding="utf-8") as f:
    line_info = json.load(f)
actual_stations = line_info[key]
print("total lines:", len(line_info.keys()))
ac_line = []
for item in actual_stations:
    ac_line.append(item[::-1])

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
min_lng = min(start_point[0], end_point[0])
min_lat = min(start_point[1], end_point[1])
max_lng = max(end_point[0], start_point[0])
max_lat = max(end_point[1], start_point[1])
print(min_lng, min_lat, max_lng, max_lat)
candidate_stops = []

expand_range = 0.15
for item in stations:
    if item[0] >= min_lng-expand_range and item[0] <= max_lng+expand_range and item[1] <= max_lat+expand_range and item[1] >= min_lat-expand_range:
        candidate_stops.append(item)
candidate_pois = []
expand_range1 = expand_range+0.001
for item in tong_points:
    if item[0] >= min_lng-expand_range1 and item[0] <= max_lng+expand_range1 and item[1] <= max_lat+expand_range1 and item[1] >= min_lat-expand_range1:
        candidate_pois.append(item)

G = nx.Graph()
nodes_tmp = list(range(len(candidate_stops)+2))
nodes = list(map(str, nodes_tmp))
edges = []

# use queue to build graph from start point to end point
q = queue.Queue()
q.put(start_point)
tags = [True]*len(candidate_stops)
times = 0
start_dis = 0.35
end_dis = 0.7
node_set = []
while not q.empty():
    ss = q.qsize()
    for i in range(ss):
        if times == 0:
            temp = q.get()
            for j in range(len(candidate_stops)):
                if candidate_stops[j][0]==end_point[0] and candidate_stops[j][1]==end_point[1]:
                    continue
                val_tmp = geodistance(temp[0], temp[1], candidate_stops[j][0], candidate_stops[j][1])
                if val_tmp <= end_dis and val_tmp >= start_dis:
                    edges.append((str(0), str(j+1)))
                    q.put(j)
                    node_set.append(j)
                    tags[j] = False
        else:
            inx = q.get()
            for j in range(len(candidate_stops)):
                if candidate_stops[j][0]==end_point[0] and candidate_stops[j][1]==end_point[1]:
                    continue
                val_tmp = geodistance(candidate_stops[inx][0], candidate_stops[inx][1], candidate_stops[j][0], candidate_stops[j][1])
                if val_tmp <= end_dis and val_tmp >= start_dis:
                    edges.append((str(inx + 1), str(j + 1)))
                    if tags[j]:
                        q.put(j)
                        node_set.append(j)
                        tags[j] = False
    times += 1

nums = 0
for j in node_set:
    val_tmp = geodistance(end_point[0], end_point[1], candidate_stops[j][0], candidate_stops[j][1])
    if val_tmp <= end_dis and val_tmp >= start_dis:
        edges.append((str(j + 1), str(len(candidate_stops) + 1)))
        nums += 1

print("neighbors of end point:", nums)
G.add_nodes_from(nodes)
G.add_edges_from(edges)
print("nodes of graph:", len(G.nodes))
print("edges of graph:", len(G.edges))
a_mat = nx.to_numpy_matrix(G)


def trans(index):
    index = int(index)
    if index == 0:
        return start_point
    elif index == len(candidate_stops)+1:
        return end_point
    return candidate_stops[int(index)-1]
            

def cal_score(point):
    pos_point = trans(point)
    num_poi = 0
    for poi in candidate_pois:
        if geodistance(poi[0], poi[1], pos_point[0], pos_point[1]) <= 0.1:
            num_poi += 1
    num_subway = 0
    if tuple(pos_point) in bus_subway_new:
        num_subway = bus_subway_new[tuple(pos_point)]
    num_duplicate = 1
    for start_point in G.neighbors(point):
        if tuple([info_pos[tuple(trans(start_point))], info_pos[tuple(pos_point)]]) in duplicate_dict:
            num_duplicate_tmp = duplicate_dict[tuple(
                [info_pos[tuple(trans(start_point))], info_pos[tuple(pos_point)]])]
        elif tuple([info_pos[tuple(pos_point)], info_pos[tuple(trans(start_point))]]) in duplicate_dict:
            num_duplicate_tmp = duplicate_dict[tuple(
                [info_pos[tuple(pos_point)], info_pos[tuple(trans(start_point))]])]
        else:
            num_duplicate_tmp = 1
        num_duplicate+=num_duplicate_tmp
    if G.degree(point)>0:
        avg_dumplicate = float(num_duplicate/G.degree(point))
    else:
        avg_dumplicate = 0
    total_score = 0.4*num_poi+0.6*num_subway-0.25*avg_dumplicate
    return total_score, num_poi, num_subway, avg_dumplicate

score_dict = dict()
for node in G.nodes:
    score_dict[node] = cal_score(node)[0]

def get_score(line):
    total_score = 0
    total_pois = 0
    total_subways = 0
    total_duplicates = 0
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
        total_score = total_score+0.4*num_poi+0.6 * num_subway-0.25*duplicate_num
    print("total pois:", total_pois)
    print("total subways:", total_subways)
    print("total duplicates:", total_duplicates)
    return float(total_score/len(line))
    

print("length of actual line:", len(actual_stations))
print("avg score of actual line:", get_score(actual_stations))

rep_mat = np.ones((a_mat.shape[0], a_mat.shape[1]))
for i in range(a_mat.shape[0]):
    for j in range(i+1, a_mat.shape[1]):
        if tuple([info_pos[tuple(trans(i))], info_pos[tuple(trans(j))]]) in duplicate_dict:
            rep_mat[i, j] = duplicate_dict[tuple([info_pos[tuple(trans(i))], info_pos[tuple(trans(j))]])]
            rep_mat[j, i] = rep_mat[i, j]
print("shape:", rep_mat.shape)

class Ant():
    def __init__(self, id, alpha, beta, distance_matrix, pheromone_matrix):
        self.id = id
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.3
        self.pheromone_matrix = pheromone_matrix
        self.distance_matrix = distance_matrix  # matrix of replication
        self.new()
        
    def criter1(self):
        line = list(map(lambda x: trans(x), self.path))
        dis_diff = list(map(lambda x: geodistance(
            start_point[0], start_point[1], x[0], x[1]), line[1:len(line)]))
        return all([dis_diff[i] < dis_diff[i+1] for i in range(len(dis_diff)-1)])

    def criter2(self):
        line = list(map(lambda x: trans(x), self.path))
        dis_diff = list(map(lambda x: geodistance(
            end_point[0], end_point[1], x[0], x[1]), line[1:len(line)]))
        return all([dis_diff[i] > dis_diff[i+1] for i in range(len(dis_diff)-1)])
    
    def criter3(self):
        line = list(map(lambda x: trans(x), self.path))
        for i in range(2, len(line)):
            lng_tmp = line[i][0]
            lat_tmp = line[i][1]
            dis_diff = np.array(
                list(map(lambda x: geodistance(lng_tmp, lat_tmp, x[0], x[1]), line[:i])))
            if np.argmin(dis_diff) != dis_diff.shape[0]-1:
                return False
        return True
    
    def new(self):
        self.path = []
        self.start_point = str(0)
        self.current_pos = str(0)
        self.end_point = str(len(nodes_tmp)-1)
        self.city_status = [True for _ in range(len(candidate_stops)+2)]
        self.path.append(self.current_pos)
        self.city_status[int(self.current_pos)] = False
        self.overall_score = 0
        self.move_count = 1
        self.total_rep = 0

    def next_city(self):
        nodes = list(G.neighbors(self.current_pos))
        num_points = G.degree(self.current_pos)
        probs = [0.0 for _ in range(num_points)]
        judge_list = [False for _ in range(num_points)]
        total_prob = 0.0
        next_city = -1
        indexes = []
        # print("next...")
        for i in range(len(nodes)):
            if self.city_status[int(nodes[i])]:
                self.path.append(nodes[i])
                if (len(self.path) >= 3 and self.criter1() and self.criter2() and self.criter3()) or (len(self.path) < 3 and self.criter1() and self.criter2()):
                    judge_list[i] = True
                    score = score_dict[nodes[i]]
                    probs[i] = pow(score, self.alpha)*pow(1.0 / self.distance_matrix[int(self.current_pos), int(nodes[i])], self.beta)*self.pheromone_matrix[int(self.current_pos), int(nodes[i])]
                    indexes.append(i)
                    total_prob += probs[i]
                self.path.pop()
        
        cur_val = random.uniform(0, 1)
        if num_points > 0 and True in judge_list:
            p_sums=np.sum(np.array(probs))
            if p_sums>0:
                if cur_val <= self.epsilon:
                    probs_new = np.array(probs)/p_sums
                    nonz_index = np.nonzero(probs_new)[0]
                    probs_new = list(probs_new[nonz_index])
                    next_city = np.random.choice(nonz_index, p=probs_new)
                else:
                    next_city = np.argsort(np.array(probs))[::-1][0]
            else:
                next_city = np.random.choice(indexes)
        
        if next_city == -1:
            tmp1 = trans(self.current_pos)
            tmp2 = trans(self.end_point)
            diss = geodistance(tmp1[0], tmp1[1], tmp2[0], tmp2[1])
            print("distance from end:", diss)
            return -1
        else:
            self.path.append(nodes[next_city])
            self.city_status[int(nodes[next_city])] = False
            self.overall_score = self.overall_score + score_dict[self.current_pos]
            self.total_rep+=self.distance_matrix[int(self.current_pos), int(nodes[next_city])]
            self.move_count += 1
            return nodes[next_city]

    def search(self):
        self.new()
        while self.current_pos != self.end_point:
            next_city = self.next_city()
            if next_city==-1:
                self.overall_score = float("-inf")
                return False
            self.current_pos = next_city
        self.overall_score = float(self.overall_score/self.move_count)
        return True


class ACO():
    def __init__(self, ant_num, max_iter, rho, Q, rep_mat, p_mat):
        self.ant_num = ant_num
        self.max_iter = max_iter
        self.distance_matrix = rep_mat
        self.pheromone_matrix = p_mat
        self.rho = rho
        self.Q = Q
        self.alpha = 2
        self.beta = 2
        self.ants = [Ant(ID, self.alpha, self.beta, rep_mat, p_mat) for ID in range(self.ant_num)]
        self.best_ant = Ant(-1, self.alpha, self.beta, rep_mat, p_mat)
        self.best_ant.overall_score = float("-inf")
        self.iter = 1

    def update_pheromone(self, jd):   # jd: num
        num_points = self.distance_matrix.shape[0]
        temp_pheromone = np.zeros((num_points, num_points))
        for ant in self.ants:
            for i in range(1, len(ant.path)):
                temp_pheromone[int(ant.path[i-1]), int(ant.path[i])] += 1
        temp_pheromone = temp_pheromone/self.ant_num
        for i in range(num_points):
            for j in range(num_points):
                if jd:
                    self.pheromone_matrix[i, j] = self.pheromone_matrix[i, j]+temp_pheromone[i, j]*self.rho
                else:
                    self.pheromone_matrix[i, j] = self.pheromone_matrix[i, j]-temp_pheromone[i, j]*self.rho
                if self.pheromone_matrix[i, j] < 0:
                    self.pheromone_matrix[i, j] = 0
        self.ants = [Ant(ID, self.alpha, self.beta, self.distance_matrix, self.pheromone_matrix)
                     for ID in range(self.ant_num)]
        
    def update_pheromone_v1(self, jd):   # jd: list
        num_points = self.distance_matrix.shape[0]
        temp_pheromone = np.zeros((num_points, num_points))
        times = 0
        for ant in self.ants:
            if not jd[times]:
                for i in range(1, len(ant.path)):
                    temp_pheromone[int(ant.path[i-1]), int(ant.path[i])] += 1
            times+=1
        
        temp_pheromone = temp_pheromone/self.ant_num
        for i in range(num_points):
            for j in range(num_points):
                self.pheromone_matrix[i, j] = self.pheromone_matrix[i, j]-temp_pheromone[i, j]*self.rho
                if self.pheromone_matrix[i, j] < 0:
                        self.pheromone_matrix[i, j] = 0
        self.ants = [Ant(ID, self.alpha, self.beta, self.distance_matrix, self.pheromone_matrix)
                     for ID in range(self.ant_num)]

    def search_path(self):
        while self.iter < self.max_iter:
            print("iter:", self.iter)
            jd = False
            for ant in self.ants:
                judge_info = ant.search()
                if judge_info and ant.overall_score > self.best_ant.overall_score:
                    print("path:", ant.path)
                    print("overall score:", ant.overall_score)
                    self.best_ant = copy.deepcopy(ant)
                    jd = True
            self.update_pheromone(jd)
            self.iter += 1
        return self.best_ant.path, self.best_ant.overall_score
    
    def search_path_v1(self):
        while self.iter < self.max_iter:
            print("iter:", self.iter)
            jd = [False]*self.ant_num
            times = 0
            for ant in self.ants:
                judge_info = ant.search()        
                if judge_info and ant.overall_score > self.best_ant.overall_score:
                    print("path:", ant.path)
                    print("overall score:", ant.overall_score)
                    self.best_ant = copy.deepcopy(ant)
                    jd[times] = True
                times+=1
            self.update_pheromone_v1(jd)
            self.iter += 1
        return self.best_ant.path, self.best_ant.overall_score

p_mat = np.ones((rep_mat.shape[0], rep_mat.shape[1]))
aco = ACO(100, 300, 0.15, 100, rep_mat, p_mat)
path, overall_score = aco.search_path_v1()
print("path:", path)
print("overall_score:", overall_score)
tp_line = []
for nd in path:
    tp_line.append(trans(nd)[::-1])
plot_line(ac_line, tp_line)

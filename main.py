import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
import osmnx as ox
import networkx as nx
import random
import math
from typing import Dict, List

# Класс AntCVRP
class AntCVRP:
    A = 1.04  # Весовой коэффициент для феромона
    B = 1.5  # Весовой коэффициент для эвристической информации
    U = 4    # Параметр для обновления феромона
    P = 0.74 # Параметр для обновления феромона

    def __init__(self, start_node: int, capacity: int, graph):
        """
        Инициализация муравья.
        :param start_node: Начальный узел.
        :param capacity: Грузоподъемность муравья.
        :param graph: Граф.
        """
        self.start_node = start_node
        self.cur_node = start_node
        self.cur_capacity = capacity
        self.max_capacity = capacity
        self.graph = graph
        self.nodes_to_visit: Dict[int, int] = {}  # Узлы, которые нужно посетить

    def init(self, nodes_to_visit: Dict[int, int]):
        """
        Инициализация муравья перед началом маршрута.
        :param nodes_to_visit: Узлы для посещения.
        """
        self.nodes_to_visit = nodes_to_visit.copy()

    def choose_next(self) -> int:
        """
        Выбор следующего узла для посещения.
        :return: Следующий узел.
        """
        max_node = 0
        total_sum = 0
        possible_nodes: Dict[int, int] = {}

        # Фильтрация узлов, которые можно посетить с учетом грузоподъемности
        for node in self.nodes_to_visit:
            if self.graph.nodes[node].get('demand', 0) <= self.cur_capacity:
                possible_nodes[node] = node
                total_sum += self.h_value(node)

        if total_sum != 0:
            # Вероятностный выбор следующего узла
            rand = random.random()
            segment = 0
            for node in possible_nodes:
                segment += self.h_value(node) / total_sum
                if rand < segment:
                    max_node = node
                    break

        if max_node != 0:
            # Удаляем узел из списка для посещения и уменьшаем грузоподъемность
            del self.nodes_to_visit[max_node]
            self.cur_capacity -= self.graph.nodes[max_node].get('demand', 0)
        else:
            # Возвращаемся на склад (начальный узел) и восстанавливаем грузоподъемность
            self.cur_capacity = self.max_capacity

        return max_node

    def h_value(self, node: int) -> float:
        """
        Вычисление значения для вероятностного выбора следующего узла.
        :param node: Узел.
        :return: Значение для выбора.
        """
        tau = self.graph.edges[self.cur_node, node].get('tau', 1.0)  # Феромон
        etha = 1.0 / self.graph.edges[self.cur_node, node].get('length', 1.0)  # Эвристическая информация
        value = math.pow(tau, self.A) * math.pow(etha, self.B)
        return value if value != 0 else float('inf')  # Возвращаем минимальное значение, если value == 0

    def local_updating_rule(self, path: List[int], length: float):
        """
        Локальное обновление феромона на пути.
        :param path: Пройденный путь.
        :param length: Длина пути.
        """
        for i in range(1, len(path)):
            curr_vertex = path[i - 1]
            next_vertex = path[i]
            # Обновление феромона на ребре (curr_vertex -> next_vertex)
            new_tau = self.graph.edges[curr_vertex, next_vertex].get('tau', 1.0) + self.U / (self.P * length)
            self.graph.edges[curr_vertex, next_vertex]['tau'] = new_tau
            # Обновление феромона на ребре (next_vertex -> curr_vertex)
            new_tau_reverse = self.graph.edges[next_vertex, curr_vertex].get('tau', 1.0) + self.U / (self.P * length)
            self.graph.edges[next_vertex, curr_vertex]['tau'] = new_tau_reverse

    def better(self, path_value1: float, path_value2: float) -> bool:
        """
        Сравнение двух путей.
        :param path_value1: Значение первого пути.
        :param path_value2: Значение второго пути.
        :return: True, если первый путь лучше.
        """
        return path_value1 < path_value2

    def end(self) -> bool:
        """
        Проверка завершения маршрута.
        :return: True, если все узлы посещены и муравей вернулся на склад.
        """
        return not self.nodes_to_visit and self.cur_node == self.start_node


# Загрузка данных
city = "Moscow"
opt = pd.read_csv("McDonalds.csv")  # Загрузка данных
opt = opt[opt["city"] == city][["city", "address", "latitude", "longitude"]].reset_index(drop=True)  # Фильтрация по городу
opt = opt.reset_index().rename(columns={"index": "id", "latitude": "y", "longitude": "x"})  # Переименование столбцов

print("Total locations:", len(opt))
print(opt.head(3))

# Подготовка данных для визуализации
data = opt.copy()
data["color"] = np.where(data['id'] == 0, 'red', 'blue')  # Начальная точка красная, остальные синие
start = data[data["id"] == 0][["y", "x"]].values[0]
print("Starting point:", start)

# Создание карты
map = folium.Map(location=start, tiles="cartodbpositron", zoom_start=12)

# Добавление маркеров на карту
for _, row in data.iterrows():
    color = "red" if row["id"] == 0 else "blue"  # Начальная точка красная, остальные синие
    folium.CircleMarker(
        location=[row["y"], row["x"]],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"ID: {row['id']}<br>Address: {row['address']}"  # Подсказка с ID и адресом
    ).add_to(map)

map

# Создание графа
G = ox.graph_from_point(start, dist=10000, network_type="drive")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
fig, ax = ox.plot_graph(G, bgcolor="black", node_size=5, node_color="white", figsize=(16, 8))

# Нахождение ближайших узлов
start_node = ox.distance.nearest_nodes(G, start[1], start[0])
opt["node"] = opt[["y", "x"]].apply(lambda x: ox.distance.nearest_nodes(G, x[1], x[0]), axis=1)
opt = opt.drop_duplicates("node", keep='first')
print(opt.head())

# Функция для вычисления расстояния
def f(a, b):
    try:
        d = nx.shortest_path_length(G, source=a, target=b, method='dijkstra', weight='travel_time')
    except:
        d = np.inf  # Используем np.inf для обозначения недостижимости
    return d

# Создание матрицы расстояний
distance_matrix = np.asarray([[f(a, b) for b in opt["node"].tolist()] for a in opt["node"].tolist()])
distance_matrix = pd.DataFrame(distance_matrix, columns=opt["node"].values, index=opt["node"].values)

print(distance_matrix.head())

# Создание тепловой карты
heatmap = distance_matrix.copy()

# Преобразование значений для визуализации
for col in heatmap.columns:
    heatmap[col] = heatmap[col].apply(lambda x:
                                      0.3 if np.isinf(x) else  # Недостижимые узлы
                                      (1 if x == 0 else 0.7)  # Нулевое расстояние или достижимые узлы
                                      )

# Визуализация тепловой карты
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, ax=ax)
plt.title("Heatmap of Travel Times")
plt.show()

# Решение задачи маршрутизации с использованием AntCVRP
drivers = 1
lst_nodes = opt["node"].tolist()
print("Start:", start_node, "Total locations to visit:", len(lst_nodes) - 1, "Drivers:", drivers)

# Инициализация муравья
ant = AntCVRP(start_node, capacity=100, graph=G)
nodes_to_visit = {node: node for node in lst_nodes if node != start_node}
ant.init(nodes_to_visit)

# Поиск маршрута
route_idx = []
while not ant.end():
    next_node = ant.choose_next()
    route_idx.append(next_node)
    ant.cur_node = next_node

print("Route for driver (nodes):", route_idx)

# Получение путей между узлами
def get_path_between_nodes(lst_route):
    lst_paths = []
    for i in range(len(lst_route)):
        try:
            a, b = lst_route[i], lst_route[i + 1]
        except:
            break
        try:
            path = nx.shortest_path(G, source=a, target=b, method='dijkstra', weight='travel_time')
            if len(path) > 1:
                lst_paths.append(path)
        except:
            continue
    return lst_paths

lst_paths = get_path_between_nodes(route_idx)

# Визуализация маршрута на карте
for path in lst_paths:
    ox.plot_route_folium(G, route=path, route_map=map, color="blue", weight=1)

map

# Создание анимации маршрута
def df_animation_multiple_path(G, lst_paths, parallel=True):
    df = pd.DataFrame()
    for path in lst_paths:
        lst_start, lst_end = [], []
        start_x, start_y = [], []
        end_x, end_y = [], []
        lst_length, lst_time = [], []

        for a, b in zip(path[:-1], path[1:]):
            lst_start.append(a)
            lst_end.append(b)
            lst_length.append(round(G.edges[(a, b, 0)]['length']))
            lst_time.append(round(G.edges[(a, b, 0)]['travel_time']))
            start_x.append(G.nodes[a]['x'])
            start_y.append(G.nodes[a]['y'])
            end_x.append(G.nodes[b]['x'])
            end_y.append(G.nodes[b]['y'])

        tmp = pd.DataFrame(list(zip(lst_start, lst_end, start_x, start_y, end_x, end_y, lst_length, lst_time)),
                           columns=["start", "end", "start_x", "start_y", "end_x", "end_y", "length", "travel_time"])
        df = pd.concat([df, tmp], ignore_index=(not parallel))

    df = df.reset_index().rename(columns={"index": "id"})
    return df

df = pd.DataFrame()
tmp = df_animation_multiple_path(G, lst_paths, parallel=False)
df = pd.concat([df, tmp], axis=0)

first_node, last_node = lst_paths[0][0], lst_paths[-1][-1]
df_start = df[df["start"] == first_node]
df_end = df[df["end"] == last_node]

fig = px.scatter_mapbox(data_frame=df, lon="start_x", lat="start_y", zoom=15, width=900, height=700,
                        animation_frame="id", mapbox_style="carto-positron")

fig.data[0].marker = {"size": 12}

fig.add_trace(px.scatter_mapbox(data_frame=opt, lon="x", lat="y").data[0])
fig.data[1].marker = {"size": 10, "color": "black"}

fig.add_trace(px.scatter_mapbox(data_frame=df_start, lon="start_x", lat="start_y").data[0])
fig.data[2].marker = {"size": 15, "color": "red"}

fig.add_trace(px.scatter_mapbox(data_frame=df_end, lon="start_x", lat="start_y").data[0])
fig.data[3].marker = {"size": 15, "color": "green"}

fig.add_trace(px.line_mapbox(data_frame=df, lon="start_x", lat="start_y").data[0])

fig.show()
from collections import deque
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random
import time
import glob

def measure_time(func):
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        result = func(*args, **kwargs)
        t2 = time.perf_counter()
        print(f'Temps d\'exécution de {func.__name__}: {t2 - t1:.2f} secondes')
        return result
    return wrapper

@measure_time
def build_graph(path):
    # Lire le fichier CSV en utilisant Pandas
    df = pd.read_csv(path, sep=';')

    # Créer un graphe vide
    graph = {}

    # Pour chaque ligne du DataFrame
    for i, row in df.iterrows():
        # Récupérer les auteurs et l'article
        authors = [author.strip().lower() for author in row['authors'].split(";")]
        article = row['articles'].strip().lower()

        # Pour chaque auteur
        for author in authors:
            # Ajouter l'auteur au graphe s'il n'y est pas déjà
            if author not in graph:
                graph[author] = []

            # Ajouter le couple (contributeur, article) à la liste de l'auteur
            for contributor in authors:
                #if author != contributor and ((contributor not in graph) or ((contributor in graph) and ((author, article) not in graph[contributor]))): #Pour avoir un undirected graph
                if author != contributor:
                    graph[author].append((contributor, article))

    # Retourner le graphe
    return graph

@measure_time
def save_graph(graph, path):
    # Obtenir toutes les clés du dictionnaire
    keys = graph.keys()

    # Créer une liste de tuples (Id, Label) à partir des clés
    data = [(i, key) for i, key in enumerate(keys)]

    # Créer un DataFrame à partir de la liste de tuples
    df_nodes = pd.DataFrame(data, columns=['Id', 'Label'])

    # Enregistrer le DataFrame
    df_nodes.to_csv(path + "nodes.csv", sep=";", index=False)

    id = {key: i for i, key in enumerate(keys)}

    # Créer une liste de tuples (Source, Target, Type, Label) à partir du dictionnaire
    data = []
    for source, edges in graph.items():
        for target, label in edges:
            data.append((id[source], id[target], 'Undirected', label))

    # Créer un DataFrame à partir de la liste de tuples
    df_edges = pd.DataFrame(data, columns=["Source", "Target", "Type", "Label"])

    # Enregistrer le DataFrame
    df_edges.to_csv(path + "edges.csv", sep=";", index=False)

def dfs(graph, node):
    visited = []
    stack = deque()
    stack.append(node)
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            unvisited = [n for n in [adjacent_list[0] for adjacent_list in graph[node]] if n not in visited]
            stack.extend(unvisited)
    return visited

@measure_time
def network_components(graph):
    size = len(graph)
    authors = list(graph.keys())
    visited = []
    components = []
    while len(visited) != size:
        random_node = random.randrange(size)
        while authors[random_node] in visited:
            random_node = random.randrange(size)
        component = dfs(graph, authors[random_node])
        visited.extend(component)
        components.append(component)
    return components

def node_distance_weight(graph, start):
    visited = set()
    queue = deque()
    queue.append(start)

    nodes = []
    node_distance_weight = {}
    node_distance_weight[start] = {'distance': 0, 'weight': 1}

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            unvisited = []
            for n, _ in graph[node]:
                if n not in visited:
                    unvisited.append(n)
                    if n not in node_distance_weight:
                        node_distance_weight[n] = {'distance': node_distance_weight[node]['distance'] + 1,
                                                   'weight': node_distance_weight[node]['weight']}
                    else:
                        if node_distance_weight[n]['distance'] == node_distance_weight[node]['distance'] + 1:
                            node_distance_weight[n]['weight'] = node_distance_weight[n]['weight'] + \
                                                                node_distance_weight[node]['weight']
            queue.extend(unvisited)
            if unvisited:
                nodes = unvisited
    return nodes, node_distance_weight

def edge_score(graph, nodes, dw):
    edge_score = {}
    visited = set()
    queue = deque()
    queue.extend(nodes)

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            for n, _ in graph[node]:
                if n not in visited:
                    if node in nodes:
                        edge_score[(node, n)] = dw[n]['weight'] / dw[node]['weight']
                    else:
                        s = 1 + sum(edge_score[edge] for edge in edge_score if node == edge[1])
                        edge_score[(node, n)] = s * (dw[n]['weight'] / dw[node]['weight'])
                    queue.append(n)
    return edge_score

def total_betweenness_all_edges(graph):
    total = {}
    for node in graph.keys():
        nodes, distance_weight = node_distance_weight(graph, node)
        edges = edge_score(graph, nodes, distance_weight)
        for edge in edges.keys():
            if edge not in total:
                total[edge] = edges[edge]
            else:
                total[edge] += edges[edge]
    return max(total, key=total.get, default=None)

@measure_time
def girvan_newman(graph):
    edge = ()
    while any(graph) and edge is not None:
        edge = total_betweenness_all_edges(graph)
        if edge is not None:
            for e in graph[edge[0]]:
                if e[0] == edge[1]:
                    graph[edge[0]].remove(e)
    return graph

def convert_graph_distance_matrix(graph):
    labels = list(graph.keys())
    dist_matrix = [[0 for x in range(len(labels))] for y in range(len(labels))]
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                if labels[j] in [x[0] for x in graph[labels[i]]]:
                    dist_matrix[i][j] = 1
    return np.array(dist_matrix), labels

@measure_time
def main(save_csv_graph=False, save_csv_community=False, save_img=False, save_dendrogram=False):
    min_component_size = 4
    max_component_size = 30
    verbose = False

    files = glob.glob("ressources/small/*.csv")
    step = 1
    for file in files:
        print("Step (%d/%d) : Processing of %s" % (step, len(files), file))
        graph = build_graph(file)
        dataset_size = "large" if "large" in file else "small"
        dataset_year = file.split("_")[1].split(".")[0]

        if save_csv_graph:
            path = f"output/{dataset_size}/csv_graph/data_{dataset_year}_"
            save_graph(graph, path)

        if save_csv_community or save_img or save_dendrogram:
            graph_community = {}
            components = network_components(graph)
            print(f"Number of components: {len(components)}")
            i = 0
            for component in components:
                component_size = len(component)
                if (component_size > min_component_size) and (component_size < max_component_size):
                    if verbose:
                        print(">>>>>>>>>> component <<<<<<<<<<")
                        print(f"Component size: {len(component)}")
                    sub_graph = {node: graph[node] for node in component if node in graph}
                    sub_graph = girvan_newman(sub_graph)
                    graph_community.update(sub_graph)
                    if save_img:
                        g = nx.Graph()
                        for key, value in sub_graph.items():
                            for v in value:
                                g.add_edge(key, v[0])
                        nx.draw_circular(g, with_labels=True)
                        path = f"output/{dataset_size}/img/{dataset_year}/{len(component)}_graph_{i}.png"
                        plt.savefig(path)
                        if verbose:
                            print(f"Image name: {len(component)}_graph_{i}.png")
                        plt.clf()
                    i += 1
            if save_csv_community:
                path = f"output/{dataset_size}/csv_community/data_{dataset_year}_"
                save_graph(graph_community, path)
            if save_dendrogram:
                path = f"output/{dataset_size}/dendrogram/dendrogram_{dataset_year}.svg"
                dist_matrix, labels = convert_graph_distance_matrix(graph_community)
                fig = ff.create_dendrogram(dist_matrix, labels=labels)
                fig.write_image(path)
        step += 1

graph = {
    'huang, xiaotao': [('qin, niannian', 'a1'), ('zhang, xiaofang', 'a2')],  # 0
    'qin, niannian': [('huang, xiaotao', 'a1'), ('wang, fen', 'a3')],  # 1
    'zhang, xiaofang': [('huang, xiaotao', 'a2'), ('wang, fen', 'a4'), ('wang, yingxu', 'a6')],  # 2
    'wang, fen': [('qin, niannian', 'a3'), ('zhang, xiaofang', 'a4'), ('peng, jun', 'a5')],  # 3
    'wang, yingxu': [('zhang, xiaofang', 'a6'), ('peng, jun', 'a7'), ('sun, zhaohao', 'a8')],  # 4
    'peng, jun': [('wang, fen', 'a5'), ('wang, yingxu', 'a7')],  # 5
    'sun, zhaohao': [('wang, yingxu', 'a8')],  # 6
    'fang, wei': [('wen, xue zhi', 'a9'), ('zheng, yu', 'a10')],  # 7
    'wen, xue zhi': [('fang, wei', 'a9')],  # 8
    'zheng, yu': [('fang, wei', 'a10')]  # 9
}

# Import files in Gephi : https://www.youtube.com/watch?v=FpOIbhOmGUs
# If Gephi bug : Just deleted the ~/Library/Application Support/gephi directory and it worked.
main(save_dendrogram=True)
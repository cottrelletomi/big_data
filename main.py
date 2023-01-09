import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import deque
import glob


def build_graph(path):
    df = pd.read_csv(path, sep=';')
    df_authors = df.assign(authors=df['authors'].str.split(";")).explode('authors')[['authors']]

    df_authors['authors'] = df_authors['authors'].str.strip()
    df_authors['authors'] = df_authors['authors'].str.lower()

    articles = []
    authors = df_authors['authors'].unique().tolist()

    map_authors = {value: key for key, value in enumerate(authors)}

    # Graph is a matrice of tuple.
    # graph = [index_author_0 : [(index_author_1, index_article_22), ...],
    #           index_author_1 : [(index_author_0, index_article_22), ...],
    #           ...
    #           index_author_n : [(index_author_47, index_article_875), ...]]
    # All index_author refer to an author in list authors
    # All index_article refer to an article in list articles

    graph = [[] for _ in range(len(authors))]

    for index, row in df.iterrows():
        articles.append(row['articles'])
        row_authors = [author.strip().lower() for author in row['authors'].split(";")]
        for author in row_authors:
            for contributor in row_authors:
                if author != contributor:
                    graph[map_authors[author]].append((map_authors[contributor], index))
                    # graph[map_authors[author]].append((map_authors[contributor], row['articles']))

    return articles, authors, graph


# Save files for Gephi
# import files : https://www.youtube.com/watch?v=FpOIbhOmGUs
def save_file(articles, authors, graph, path, undirected=False):
    # Save nodes
    df_nodes = pd.DataFrame(enumerate(authors), columns=["Id", "Label"])
    df_nodes.to_csv(path + "nodes.csv", sep=";", index=False)

    # Save edges (Directed graph) -> 73128
    edges = []
    for index, row in enumerate(graph):
        for column in row:
            edges.append((index, column[0], "Directed", articles[column[1]]))
    df_edges = pd.DataFrame(edges, columns=["Source", "Target", "Type", "Label"])
    df_edges.to_csv(path + "edges.csv", sep=";", index=False)

    # Save edges (Undirected graph) -> 36564
    if undirected:
        edges = []
        edges_undirected = []
        for _, row in df_edges.iterrows():
            if (row["Target"], row["Source"]) not in edges:
                edges.append((row["Source"], row["Target"]))
                edges_undirected.append((row["Source"], row["Target"], "Undirected", row["Label"]))
        df_edges = pd.DataFrame(edges_undirected, columns=["Source", "Target", "Type", "Label"])
        df_edges.to_csv(path + "edges_undirected.csv", sep=";", index=False)


# Distribution en degrés
def distribution_in_degrees(graph):
    x = [len(adjacent_list) for adjacent_list in graph]
    MIN, MAX = min(x), 25
    plt.hist(x, bins=range(MIN, MAX), color='blue', edgecolor='white')
    plt.xlabel('Degré des sommets')
    plt.ylabel('Nombres de sommets')
    plt.title('Distribution des degrés')
    plt.show()


# distribution_in_degrees(graph)

# Moyenne des plus courts chemins
def bfs(graph, start):
    size = len(graph)
    shortest_path = [0 for _ in range(len(graph))]  # np.zeros(size)
    mark = np.full(size, False)

    visited = []
    queue = deque()
    queue.append(start)
    level = 1

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            unvisited = [n for n in [adjacent_list[0] for adjacent_list in graph[node]] if n not in visited]
            queue.extend(unvisited)
            for i in unvisited:
                if not mark[i]:
                    shortest_path[i] = level
                    mark[i] = True
            level += 1

    return shortest_path


def matrix_shortest_path(graph):
    size = len(graph)
    matrix = np.empty([size, size])
    for i in range(0, size):
        # if i == 6000: print("1/2")
        matrix[i] = bfs(graph, i)
    return matrix


def average_shortest_path_length(graph):
    size = len(graph)
    matrix = matrix_shortest_path(graph)
    upper_sum = np.triu(matrix).sum() - np.trace(matrix)
    lower_sum = np.tril(matrix).sum() - np.trace(matrix)
    return (2 / (size * (size - 1))) * upper_sum


# print(average_shortest_path_length(graph))
# moyenne: 6.950019747204883

# Nombre de composantes du réseau
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


def network_components(graph):
    size = len(graph)
    visited = []
    components = []
    while len(visited) != size:
        random_node = random.randrange(size)
        while random_node in visited:
            random_node = random.randrange(size)
        component = dfs(graph, random_node)
        visited.extend(component)
        components.append(component)
    return components

# components = network_components(graph)
# print(len(components))

# Diamètre du réseau
# matrix = matrix_shortest_path(graph)
# print(np.amax(matrix))


# Est-il scale-free ? Si oui, quelle est la valeur de l'exposant


# NEXT --> edges betweenness : Nombre de plus court chemin passant à travers une arête
def node_distance_weight(graph, start):
    visited = []
    queue = deque()
    queue.append(start)

    nodes = []
    node_distance_weight = [{} for _ in range(len(graph))]
    node_distance_weight[start] = {'distance': 0, 'weight': 1}

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            unvisited = []
            for n in graph[node]:
                if n[0] not in visited:
                    unvisited.append(n[0])
                    if 'distance' not in node_distance_weight[n[0]]:
                        node_distance_weight[n[0]] = {'distance': node_distance_weight[node]['distance'] + 1,
                                                      'weight': node_distance_weight[node]['weight']}
                    else:
                        if node_distance_weight[n[0]]['distance'] == node_distance_weight[node]['distance'] + 1:
                            node_distance_weight[n[0]]['weight'] = node_distance_weight[n[0]]['weight'] + \
                                                                   node_distance_weight[node]['weight']
                        else:
                            pass
                            # print('do nothing')
            queue.extend(unvisited)
            if unvisited:
                nodes = unvisited
    return nodes, node_distance_weight


def edge_score(graph, nodes, dw):
    edge_score = {}
    visited = []
    queue = deque()
    queue.extend(nodes)

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.append(node)
            unvisited = []
            for n in graph[node]:
                if n[0] not in visited:
                    unvisited.append(n[0])
                    if node in nodes:
                        edge_score[(node, n[0])] = dw[n[0]]['weight'] / dw[node]['weight']
                    else:
                        s = 1 + sum([edge_score[edge] for edge in edge_score.keys() if node == edge[1]])
                        edge_score[(node, n[0])] = s * (dw[n[0]]['weight'] / dw[node]['weight'])
            queue.extend(unvisited)
    return edge_score


def edge_betweenness_centrality(graph, start):
    nodes, distance_weight = node_distance_weight(graph, start)
    return edge_score(graph, nodes, distance_weight)


def total_betweenness_all_edges(graph):
    size = len(graph)
    total = {}
    for node in range(size):
        edges = edge_betweenness_centrality(graph, node)
        for edge in edges.keys():
            if edge not in total:
                total[edge] = edges[edge]
            else:
                total[edge] += edges[edge]
    return max(total, key=total.get, default=None)


def girvan_newman(graph):
    edge = ()
    while any(graph) and edge is not None:
        edge = total_betweenness_all_edges(graph)
        if edge is not None:
            for e in graph[edge[0]]:
                if e[0] == edge[1]:
                    graph[edge[0]].remove(e)
            # print("New graph :")
            # print(g)
            # print("Edge removed :")
            # print(edge)
    return graph


# graph = [
#     [(1, 'a1'), (2, 'a2')],
#     [(0, 'a1'), (3, 'a3')],
#     [(0, 'a2'), (3, 'a4'), (4, 'a6')],
#     [(1, 'a3'), (2, 'a4'), (5, 'a5')],
#     [(2, 'a6'), (5, 'a7'), (6, 'a8')],
#     [(3, 'a5'), (4, 'a7')],
#     [(4, 'a8')]
# ]

# print(graph)
# print(girvan_newman(graph))
# save_file(girvan_newman(graph))

# If Gephi bug : Just deleted the ~/Library/Application Support/gephi directory and it worked.
def main(save=True, run_girvan_newman=False):
    files = glob.glob("ressources/small/*.csv")

    dataframes = []
    articles = []
    authors = []
    graphs = []

    step = 1
    for file in files:
        print("Step (%d/%d) : Processing of %s" % (step, len(files), file))
        dataframes.append(pd.read_csv(file, sep=';'))
        articles_buffer, authors_buffer, graph_buffer = build_graph(file)
        articles.append(articles_buffer)
        authors.append(authors_buffer)
        graphs.append(graph_buffer)
        print("> Dataset with %d authors and %d articles" % (len(authors_buffer), len(articles_buffer)))
        if run_girvan_newman:
            graph_buffer = girvan_newman(graph_buffer)
        if save:
            name = file.replace("ressources", "output").replace(".csv", "_")
            save_file(articles_buffer, authors_buffer, graph_buffer, name)
        step += 1


main(True, True)
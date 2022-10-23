import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import deque

df = pd.read_csv('ressources/data_2017.csv', sep =';')
df_authors = df.assign(authors=df['authors'].str.split(";")).explode('authors')[['authors']]


df_authors['authors'] = df_authors['authors'].str.strip()
df_authors['authors'] = df_authors['authors'].str.lower()

articles = []
authors = df_authors['authors'].unique().tolist()

map_authors = {value:key for key, value in enumerate(authors)}

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
                #graph[map_authors[author]].append((contributor, row['articles']))

#Distribution en degrés
def distribution_in_degrees():
    x = [len(adjacent_list) for adjacent_list in graph]
    min, max = min(x), 40
    plt.hist(x, bins=range(min, max), color = 'blue', edgecolor = 'white')
    plt.xlabel('Degré des sommets')
    plt.ylabel('Nombres de sommets')
    plt.title('Distribution des degrés')
    plt.show()
#distribution_in_degrees()

#Moyenne des plus courts chemins
def bfs(graph, start):
    size = len(graph)
    shortest_path = np.zeros(size) #[0 for _ in range(len(graph))]
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
        if i == 6000:
            print("1/2")
        matrix[i] = bfs(graph, i)
    return matrix

def average_shortest_path_length(graph):
    return None

#Nombre de composantes du réseau
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

def number_of_network_components(graph):
    size = len(graph)
    visited = []
    nb = 0
    while len(visited) != size:
        random_node = random.randrange(size)
        while random_node in visited:
            random_node = random.randrange(size)
        visited.extend(dfs(graph, random_node))
        nb += 1
    return nb
#print(number_of_network_components(graph))

#Diamètre du réseau
matrix = matrix_shortest_path(graph)
print(np.amax(matrix))


#Est-il scale-free ? Si oui, quelle est la valeur de l'exposant


# NEXT --> edges betweenness : Nombre de plus court chemin passant à travers une arête
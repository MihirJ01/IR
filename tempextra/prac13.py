import networkx as nx

# Create a directed graph (replace this with your own graph)
G = nx.DiGraph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 1)])

# Calculate PageRank
pagerank_scores = nx.pagerank(G)

# Calculate HITS (Hub and Authority) scores
hits_scores = nx.hits(G)

# Print the results
print("PageRank Scores:", pagerank_scores)
print("Hub Scores:", hits_scores[0])
print("Authority Scores:", hits_scores[1])

import os
import networkx as nx
import matplotlib.pyplot as plt
from xml.dom.minidom import parse

# Check if the XML file exists before parsing
xml_file = "movies.xml"

if not os.path.exists(xml_file):
    print(f"Error: '{xml_file}' not found. Make sure the file is in the correct directory.")
else:
    # Open XML document using minidom parser
    DOMTree = parse(xml_file)
    collection = DOMTree.documentElement

    # Print root element attribute if available
    if collection.hasAttribute("shelf"):
        print("Root element: %s" % collection.getAttribute("shelf"))

    # Get all the movies in the collection
    movies = collection.getElementsByTagName("movie")

    # Print details of each movie
    for movie in movies:
        print("\n***** Movie *****")
        if movie.hasAttribute("title"):
            print("Title: %s" % movie.getAttribute("title"))

        # Extract and print child elements safely
        def get_text(tag_name):
            node = movie.getElementsByTagName(tag_name)
            return node[0].childNodes[0].data.strip() if node else "N/A"

        print("Type:", get_text("type"))
        print("Format:", get_text("format"))
        print("Rating:", get_text("rating"))
        print("Description:", get_text("description"))

# Function to generate and visualize a graph
def GenerateGraph():
    G = nx.Graph()

    # Adding edges (nodes will be automatically added)
    G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")])

    # Draw the graph with improved styling
    plt.figure(figsize=(6, 4))
    nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray",
            node_size=2000, font_size=12, font_weight="bold")

    # Save and display the graph
    plt.savefig("simple_graph.png")
    plt.show()

    print("\nNodes of graph:", G.nodes())
    print("Edges of graph:", G.edges())

# Call the function to generate and display the graph
GenerateGraph()

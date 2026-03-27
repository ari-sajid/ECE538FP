import osmnx as ox

# Coordinates for the centers of EWR and LGA
airports = {
    "ewr": (40.6895, -74.1745),
    "lga": (40.7769, -73.8740)
}

for name, coords in airports.items():
    # Downloads the taxiway and runway network from OpenStreetMap
    G = ox.graph_from_point(coords, dist=3000, network_type='all')
    ox.save_graphml(G, filepath=f"data/geo/{name}_layout.graphml")
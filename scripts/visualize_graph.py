import matplotlib.pyplot as plt
import networkx as nx
import os
import json
from temporal_graph import TemporalSceneGraph

def visualize():
    # 1. Load Data
    tsg = TemporalSceneGraph()
    if not os.path.exists("event_log.json"):
        print("Error: event_log.json not found. Please run query_engine.py or temporal_graph.py first to generate it.")
        return

    tsg.load_from_json("event_log.json")
    
    # 2. Get Full Graph
    full_graph = tsg.graph
    
    # 3. Prune Data
    query = "Where is the laptop?"
    pruned_result = tsg.prune_and_retrieve(query)
    
    # Build a subgraph from pruned events just for visualization purposes
    # We want to know which nodes and edges are "active"
    active_edges = []
    active_nodes = set()
    
    for ev in pruned_result.events:
        u, v = ev.get('subject'), ev.get('object')
        if u and v:
            active_nodes.add(u)
            active_nodes.add(v)
            # Find the specific edge key/index if multiple edges exist, 
            # but for viz we can just highlight *an* edge between u and v
            active_edges.append((u, v))
            
    # 4. Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
import sys
# Allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.temporal_graph import TemporalSceneGraph

# Load Graph
# Point to one of the rich logs we generated
log_path = os.path.join(os.path.dirname(__file__), "../outputs/logs/test_video_events.json") 
if not os.path.exists(log_path):
    print(f"Log not found at {log_path}")
    # Fallback to legacy if exists
    log_path = os.path.join(os.path.dirname(__file__), "../data/event_log.json")

tsg = TemporalSceneGraph()
if os.path.exists(log_path):
    tsg.load_from_json(log_path)
    print(f"Loaded graph from {log_path} with {len(tsg.all_events)} events.")
else:
    print("No event log found.")
    exit()

# Run Query
query = "When did he use the laptop?"
result = tsg.prune_and_retrieve(query)
pruned_events = result.events

print(f"Original Graph: {tsg.graph.number_of_nodes()} nodes, {tsg.graph.number_of_edges()} edges")
print(f"Pruned Subgraph: {len(pruned_events)} events kept")

# Visualization Logic
# 1. Full Graph Layout
pos = nx.spring_layout(tsg.graph, k=0.5, seed=42)

# Figure setup
plt.figure(figsize=(20, 10))

# Subplot 1: Full Graph (Faint)
plt.subplot(1, 2, 1)
plt.title("Full Video Temporal Graph")
nx.draw_networkx(tsg.graph, pos, with_labels=True, 
                 node_color='lightgray', edge_color='lightgray', 
                 node_size=500, font_size=8, alpha=0.5, arrows=True)

# Subplot 2: Pruned Subgraph (Highlighted)
plt.subplot(1, 2, 2)
plt.title(f"Pruned Subgraph (Query: '{query}')")

# Draw full graph as background (very faint)
nx.draw_networkx(tsg.graph, pos, with_labels=False, 
                 node_color='lightgray', edge_color='lightgray', 
                 node_size=300, alpha=0.2, arrows=False)

# Identify Pruned Nodes/Edges
pruned_nodes = set()
pruned_edges = []

for ev in pruned_events:
    subj = ev.get('subject')
    obj = ev.get('object')
    if subj: pruned_nodes.add(subj)
    if obj: pruned_nodes.add(obj)
    if subj and obj:
        pruned_edges.append((subj, obj))

# Draw Pruned Elements
nx.draw_networkx_nodes(tsg.graph, pos, nodelist=list(pruned_nodes), node_color='salmon', node_size=700)
nx.draw_networkx_labels(tsg.graph, pos, labels={n:n for n in pruned_nodes}, font_size=10, font_weight='bold')
nx.draw_networkx_edges(tsg.graph, pos, edgelist=pruned_edges, edge_color='red', width=2.0)

output_path = os.path.join(os.path.dirname(__file__), "../outputs/figure_1_teaser.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Saved visualization to {output_path}")

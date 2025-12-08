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
    
    # Layout - Compute consistent layout for both plots
    # spring_layout is good, k=distance param
    pos = nx.spring_layout(full_graph, seed=42, k=0.5) 
    
    node_color = 'skyblue'
    node_size = 2000
    font_size = 10
    
    # --- Plot 1: Full Graph ---
    ax1 = axes[0]
    ax1.set_title("Full Video Temporal Graph", fontsize=14, fontweight='bold')
    
    nx.draw_networkx_nodes(full_graph, pos, ax=ax1, node_color='lightgray', node_size=node_size, alpha=0.9)
    nx.draw_networkx_labels(full_graph, pos, ax=ax1, font_size=font_size)
    nx.draw_networkx_edges(full_graph, pos, ax=ax1, edge_color='gray', width=1.0, alpha=0.5, arrows=True)
    
    # Annotate edge info (optional, might get cluttery, let's keep it simple structure)
    
    ax1.axis('off')
    
    # --- Plot 2: Pruned Subgraph ---
    ax2 = axes[1]
    ax2.set_title(f"Pruned Subgraph (Query: 'Laptop')", fontsize=14, fontweight='bold')
    
    # Draw faint background
    nx.draw_networkx_nodes(full_graph, pos, ax=ax2, node_color='lightgray', node_size=node_size, alpha=0.2)
    nx.draw_networkx_edges(full_graph, pos, ax=ax2, edge_color='lightgray', width=1.0, alpha=0.2, arrows=False)
    
    # Draw Active Elements
    if active_nodes:
        nx.draw_networkx_nodes(full_graph, pos, ax=ax2, nodelist=list(active_nodes), node_color='salmon', node_size=node_size, alpha=1.0)
        nx.draw_networkx_labels(full_graph, pos, ax=ax2, labels={n: n for n in active_nodes}, font_size=font_size, font_weight='bold')
        
    if active_edges:
        nx.draw_networkx_edges(full_graph, pos, ax=ax2, edgelist=active_edges, edge_color='red', width=2.0, alpha=1.0, arrows=True)

    ax2.axis('off')
    
    # Save
    output_file = "figure_1_teaser.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Successfully saved visualization to {output_file}")

if __name__ == "__main__":
    visualize()

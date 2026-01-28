import networkx as nx
import json
import os

class QueryResult:
    def __init__(self, events, compression_ratio):
        self.events = events
        self.compression_ratio = compression_ratio

class TemporalSceneGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.all_events = []

    def load_from_json(self, path):
        """Loads events from a JSON file and builds the graph."""
        with open(path, 'r') as f:
            self.all_events = json.load(f)

        for event in self.all_events:
            subj = event.get('subject')
            obj = event.get('object')
            
            # Add nodes if they don't exist
            if subj:
                self.graph.add_node(subj)
            if obj:
                self.graph.add_node(obj)
            
            # Add edge if both exist
            if subj and obj:
                self.graph.add_edge(
                    subj, 
                    obj, 
                    timestamp=event.get('timestamp'),
                    interaction_type=event.get('type'),
                    frame=event.get('frame'),
                    raw_event=event # Store the raw event for easy retrieval
                )

    def prune_and_retrieve(self, user_query, config=None):
        """
        Retrieves events based on the user query and configuration.
        """
        if config is None:
            config = {}
        
        pruning_cfg = config.get('graph', {}).get('pruning', {})
        hop_depth = pruning_cfg.get('hop_depth', 1)
        top_k = pruning_cfg.get('top_k_neighbors', 5) # Per anchor
        # threshold = pruning_cfg.get('jaccard_threshold', 0.1) # Unused if doing strict anchor expansion
        
        user_query_lower = user_query.lower()
        
        # Helper to extract class from node ID (e.g., "person-1" -> "person")
        def get_class(node_name):
            parts = node_name.split('-')
            if len(parts) > 1:
                return "-".join(parts[:-1]).lower()
            return None

        # Step 1: Identify Anchors
        anchors = set()
        
        # 1.1 Exact ID match (Optional but good)
        for node in self.graph.nodes():
            if node.lower() in user_query_lower: # Simple substring check
                 anchors.add(node)
        
        # 1.2 Class Semantic Match (Simple substring for now, or embeddings later)
        if not anchors:
             # Fallback to seeking any nodes whose class is in query
             for node in self.graph.nodes():
                 cls = get_class(node)
                 if cls and cls in user_query_lower:
                     anchors.add(node)
        
        # Step 2: Expansion (Multi-hop)
        relevant_nodes = set(anchors)
        current_frontier = set(anchors)
        
        for hop in range(hop_depth):
            next_frontier = set()
            for node in current_frontier:
                # Outgoing edges
                out_edges = self.graph.out_edges(node, data=True)
                # Incoming edges
                in_edges = self.graph.in_edges(node, data=True)
                
                # Combine neighbors
                neighbors = []
                for u, v, data in out_edges:
                    neighbors.append((v, data))
                for u, v, data in in_edges:
                    neighbors.append((u, data))
                
                # Sort neighbors by confidence/interaction strength if available
                # Assuming data['raw_event'].get('confidence', 1.0)
                neighbors.sort(key=lambda x: x[1].get('raw_event', {}).get('confidence', 0.0), reverse=True)
                
                # Top-K
                subset = neighbors[:top_k]
                for neigh_node, _ in subset:
                    if neigh_node not in relevant_nodes:
                        relevant_nodes.add(neigh_node)
                        next_frontier.add(neigh_node)
            
            current_frontier = next_frontier
            if not current_frontier:
                break
        
        # Step 3: Collect Events
        # Retrieve all edges between relevant nodes
        relevant_events = []
        
        # We want edges where BOTH or AT LEAST ONE? 
        # Usually subgraph induced by valid nodes. 
        # But for "Expansion", we usually want all edges TOUCHING the subgraph.
        # Let's collect edges connected to relevant nodes (which includes the frontier).
        
        # Just grab edges in the subgraph induced by relevant nodes
        # subgraph = self.graph.subgraph(relevant_nodes)
        # for u, v, data in subgraph.edges(data=True):
        #      if 'raw_event' in data:
        #           relevant_events.append(data['raw_event'])
                   
        # Actually, strict subgraph might miss "person-1 interacting with unrelated-object" if unrelated-object wasn't picked.
        # But if hop logic works, they should be picked.
        # Let's stick to edges connected to any node in relevant_nodes, but duplicated check needed.
        
        seen_events = set()
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if u in relevant_nodes or v in relevant_nodes:
                if 'raw_event' in data:
                    # Dedupe
                    ev = data['raw_event']
                    ev_str = json.dumps(ev, sort_keys=True)
                    if ev_str not in seen_events:
                        relevant_events.append(ev)
                        seen_events.add(ev_str)
        
        # Fallback if empty?
        if not relevant_events and not anchors:
            # Full retrieval or fallback
            return QueryResult([], 0.0)

        # Sort by timestamp
        relevant_events.sort(key=lambda x: x['timestamp'])

        # Calculate efficiency
        total_events = len(self.all_events)
        retrieved_count = len(relevant_events)
        
        if total_events > 0:
            compression_ratio = 1.0 - (retrieved_count / total_events)
        else:
            compression_ratio = 0.0

        return QueryResult(relevant_events, compression_ratio)

if __name__ == "__main__":
    # Create dummy event_log.json
    dummy_data = [
        {
            "timestamp": 120.5,
            "frame": 3600,
            "type": "START",
            "subject": "person-1",
            "object": "coffee_cup-3"
        },
        {
            "timestamp": 125.0,
            "frame": 3750,
            "type": "END",
            "subject": "person-1",
            "object": "coffee_cup-3"
        },
        {
            "timestamp": 400.2,
            "frame": 12000,
            "type": "START",
            "subject": "person-1",
            "object": "laptop-7"
        }
    ]
    
    dummy_filename = "dummy_event_log.json"
    with open(dummy_filename, "w") as f:
        json.dump(dummy_data, f, indent=2)
    
    try:
        # Initialize graph logic
        tsg = TemporalSceneGraph()
        tsg.load_from_json(dummy_filename)
        
        # Test Query
        query = "When did he use the laptop?"
        print(f"Query: {query}")
        
        result = tsg.prune_and_retrieve(query)
        
        print(f"\nPruned Events ({len(result.events)}):")
        print(json.dumps(result.events, indent=2))
        
        print(f"\nEfficiency Gain: {result.compression_ratio:.0%}")
        
    finally:
        # Cleanup
        if os.path.exists(dummy_filename):
            os.remove(dummy_filename)

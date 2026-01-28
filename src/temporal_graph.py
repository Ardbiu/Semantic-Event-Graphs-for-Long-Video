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
        query_tokens = set(user_query_lower.split())
        
    def prune_and_retrieve(self, user_query, threshold=0.1, config=None):
        """
        Prune graph to retrieve relevant subgraph for the query.
        """
        user_query_lower = user_query.lower()
        
        # Configurable params
        hop_depth = 1
        top_k = 5
        if config:
            hop_depth = config.get('graph', {}).get('pruning', {}).get('hop_depth', 1)
            top_k = config.get('graph', {}).get('pruning', {}).get('top_k_neighbors', 5)

        # 1. Anchor Identification
        anchors = set()
        for node in self.graph.nodes():
            # Simple lexical matching for anchors
            node_label = str(node).lower() 
            if node_label in user_query_lower:
                anchors.add(node)
                
        relevant_nodes = set(anchors)
        
        # 2. Expansion (Multi-hop)
        current_frontier = set(anchors)
        
        for hop in range(hop_depth):
            next_frontier = set()
            for node in current_frontier:
                # Get neighbors with edge data
                # We want to identify neighbors with strongest "Interaction Confidence"
                # Edge(u, v) -> data['max_confidence'] ideally, or aggregate from events
                
                neighbors = []
                # Check outgoing
                if self.graph.has_node(node):
                    for nbr in self.graph.neighbors(node):
                        # Calculate scores
                        score = 0.0
                        # Multigraph: multiple edges possible
                        edge_data = self.graph.get_edge_data(node, nbr)
                        # NetworkX MultiDiGraph get_edge_data returns {key: {attr}}
                        if edge_data:
                            for key, attrs in edge_data.items():
                                ev = attrs.get('raw_event', {})
                                conf = ev.get('confidence', 0.0)
                                if conf > score:
                                    score = conf
                        neighbors.append((nbr, score))
                
                # Sort by score
                neighbors.sort(key=lambda x: x[1], reverse=True)
                
                # Top k
                subset = neighbors[:top_k]
                for nbr, _ in subset:
                    if nbr not in relevant_nodes:
                        relevant_nodes.add(nbr)
                        next_frontier.add(nbr)
            
            current_frontier = next_frontier
            if not current_frontier:
                break
                
        # 3. Subgraph Extraction & Event Collection
        subgraph = self.graph.subgraph(relevant_nodes)
        
        # Collect unique events attached to these edges
        unique_events = {} # key -> event dict
        
        for u, v, k, data in subgraph.edges(keys=True, data=True):
             if 'raw_event' in data:
                 ev = data['raw_event']
                 # Dedupe by timestamp/subject/object hash or just ID if we had one.
                 # Using timestamp+subj+obj as unique key
                 key_str = f"{ev.get('timestamp')}_{ev.get('subject')}_{ev.get('object')}_{ev.get('type')}"
                 unique_events[key_str] = ev

        relevant_events = list(unique_events.values())
        relevant_events.sort(key=lambda x: x['timestamp'])

        # Compression calculation
        total_events = len(self.all_events) or 1
        compression_ratio = 1.0 - (len(relevant_events) / total_events)

        class PrunedResult:
            def __init__(self, events, ratio):
                self.events = events
                self.compression_ratio = ratio
                
        return PrunedResult(relevant_events, compression_ratio)

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

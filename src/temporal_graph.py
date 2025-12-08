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

    def prune_and_retrieve(self, user_query, threshold=0.1):
        """
        Retrieves events based on the user query.
        
        Step 1 (Anchors): specific objects in the query.
        Step 2 (Expansion): Find all edges connected to those anchor nodes.
        Step 3 (Fallback): Jaccard similarity.
        """
        user_query_lower = user_query.lower()
        query_tokens = set(user_query_lower.split())
        
        # Step 1: Identify Anchors
        # Step 1: Identify Anchors
        anchors = []
        
        # Helper to extract class from node ID (e.g., "person-1" -> "person")
        def get_class(node_name):
            parts = node_name.split('-')
            if len(parts) > 1:
                return "-".join(parts[:-1]).lower()
            return None

        specific_matches = []
        generic_matches = []
        classes_with_specifics = set()

        for node in self.graph.nodes():
            node_lower = node.lower()
            
            # Check for EXACT (Specific) Match of the ID
            # We want to ensure we match "person-1" but not "person-10" if query is "person-1"
            # Simple substring check 'node.lower() in user_query_lower' is risky for "person-1" inside "person-10"
            # But standard tokenization is better.
            # Let's stick to the current string check but be mindful, or use token boundaries if needed.
            # Given the loop is over graph nodes, if "person-10" is in graph, and query has "person-10", it matches.
            
            if node_lower in user_query_lower:
                specific_matches.append(node)
                cls = get_class(node)
                if cls:
                    classes_with_specifics.add(cls)
            else:
                # Check for Generic Class Match
                cls = get_class(node)
                if cls and cls in user_query_lower:
                    generic_matches.append(node)

        # Merge Phase:
        # 1. Always include specific matches
        anchors.extend(specific_matches)
        
        # 2. Include generic matches ONLY if their class was NOT covered by a specific match
        for node in generic_matches:
            cls = get_class(node)
            if cls not in classes_with_specifics:
                anchors.append(node)
        
        relevant_events = []
        
        if anchors:
            # Step 2: Expansion
            # Find edges connected to anchors
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                if u in anchors or v in anchors:
                    if 'raw_event' in data:
                        relevant_events.append(data['raw_event'])
        else:
            # Step 3: Fallback (Jaccard similarity)
            for event in self.all_events:
                # Construct a string representation of the event
                event_str = f"{event.get('type', '')} {event.get('subject', '')} {event.get('object', '')}".lower()
                event_tokens = set(event_str.split())
                
                intersection = query_tokens.intersection(event_tokens)
                union = query_tokens.union(event_tokens)
                
                if not union:
                    jaccard_score = 0.0
                else:
                    jaccard_score = len(intersection) / len(union)
                
                if jaccard_score >= threshold:
                    relevant_events.append(event)

        # Deduplicate events based on content (simple approach: use json string representation)
        seen = set()
        unique_events = []
        for ev in relevant_events:
            ev_str = json.dumps(ev, sort_keys=True)
            if ev_str not in seen:
                seen.add(ev_str)
                unique_events.append(ev)
        
        relevant_events = unique_events

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

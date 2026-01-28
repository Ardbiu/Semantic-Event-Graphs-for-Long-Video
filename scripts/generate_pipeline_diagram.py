import graphviz
import os

def main():
    # Create the graph
    dot = graphviz.Digraph('pipeline', comment='Pipeline Diagram')
    
    # Global Attributes
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', fontname='Helvetica', style='filled', fillcolor='white')
    dot.attr('edge', fontname='Helvetica')
    
    # --- Input Node ---
    dot.node('Input', 'Raw Video Input', shape='parallelogram', fillcolor='lightgrey')
    
    # --- Visual Processing Subgraph ---
    with dot.subgraph(name='cluster_VisualProcessing') as c:
        c.attr(label='Visual Processing', fontname='Helvetica', style='rounded', color='black')
        c.node('ModuleA', 'Object Detection & Tracking\n(YOLOv11)')
        c.node('ModuleB', 'Event Extraction\n(Proximity-based)', xlabel='Output: START/END Logs')
        c.edge('ModuleA', 'ModuleB')
        
    # --- Graph Reasoning Subgraph ---
    with dot.subgraph(name='cluster_GraphReasoning') as c:
        c.attr(label='Graph Reasoning', fontname='Helvetica', style='rounded', color='black')
        c.node('ModuleC', 'Temporal Scene Graph\n(TSG)')
        c.node('ModuleD', 'Query-Aware Pruning')
        # We don't connect C->D here yet, we do it in the main graph to handle flow
        
    # --- Other Nodes ---
    dot.node('UserQuery', 'User Query', shape='parallelogram', fillcolor='lightgrey')
    dot.node('ModuleE', 'LLM Reasoning\n(Gemini 2.5 Flash)')
    dot.node('Output', 'Final Answer', shape='parallelogram', fillcolor='lightgrey')
    
    # --- Main Flow Connections ---
    dot.edge('Input', 'ModuleA')
    dot.edge('ModuleB', 'ModuleC')
    dot.edge('ModuleC', 'ModuleD')
    dot.edge('UserQuery', 'ModuleD')
    dot.edge('ModuleD', 'ModuleE')
    dot.edge('ModuleE', 'Output')
    
    # --- Save ---
    output_dir = os.path.join(os.path.dirname(__file__), "../outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pipeline_fig1")
    
    # Render
    try:
        dot.render(output_path, format='png', cleanup=True)
        print(f"Successfully generated diagram at {output_path}.png")
    except Exception as e:
        print(f"Error generating diagram: {e}")
        print("Make sure Graphviz binaries are installed on your system.")
        print("Mac: brew install graphviz")
        print("Linux: sudo apt-get install graphviz")

if __name__ == "__main__":
    main()

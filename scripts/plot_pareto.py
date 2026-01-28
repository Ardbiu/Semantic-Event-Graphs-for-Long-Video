import matplotlib.pyplot as plt
import os

def main():
    # Data Points
    # Strategy: (Tokens, Accuracy, Color, Marker)
    data = {
        "Short-Context": (1030, 2.5, "gray", "o"),
        "Long-Context": (40390, 62.5, "blue", "o"),
        "HyperGraph-V (Ours)": (3466, 65.0, "red", "*")
    }

    # Setup Plot
    plt.figure(figsize=(10, 7))
    
    # Plot Points
    for name, (tokens, acc, color, marker) in data.items():
        # Size multiplier: make "Ours" star bigger
        size = 300 if marker == "*" else 150
        
        plt.scatter(tokens, acc, color=color, marker=marker, s=size, label=name, edgecolors='black', zorder=10)
        
        # Annotate
        xytext_offset = (0, 10)
        ha_align = 'center'
        
        # Adjust specific labels to avoid clutter
        if "Short" in name:
            xytext_offset = (20, 10)
            ha_align = 'left'
        elif "Long" in name:
            xytext_offset = (-20, -20)
            ha_align = 'right'
        
        plt.annotate(
            f"{name}\n({tokens:,} tokens, {acc}%)",
            (tokens, acc),
            xytext=xytext_offset, 
            textcoords='offset points',
            ha=ha_align, 
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9)
        )

    # Aesthetics
    plt.title("Accuracy vs Token Usage", fontsize=16, fontweight='bold')
    plt.xlabel("Input Tokens (Lower is Better)", fontsize=13)
    plt.ylabel("Accuracy (%) (Higher is Better)", fontsize=13)
    
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    # Axes Limits
    plt.xlim(-2000, 45000)
    plt.ylim(0, 100)
    
    # Highlight "Corner of Success" (Top Left)
    # Optional shading or arrow
    # Let's draw an arrow from Long to HyperGraph to show efficiency gain
    start_pos = data["Long-Context"][:2]
    end_pos = data["HyperGraph-V (Ours)"][:2]
    
    plt.annotate(
        "90% Efficiency Gain",
        xy=end_pos, xytext=(25000, 40),
        arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=10),
        fontsize=12, color='green', fontweight='bold'
    )

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "../outputs/figure_2_accuracy_vs_tokens.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Figure 2 to {output_path}")

if __name__ == "__main__":
    main()

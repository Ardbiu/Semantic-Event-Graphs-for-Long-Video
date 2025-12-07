"""
CLI for building a graph memory JSON file from YOLO detections.
"""

from __future__ import annotations

import argparse

from graph_memory import build_graph_memory_from_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GraphMemory from detections.")
    parser.add_argument(
        "--detections",
        required=True,
        help="Path to the detection JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the GraphMemory JSON will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_memory, num_frames = build_graph_memory_from_detections(
        json_path=args.detections, output_path=args.output
    )

    num_tracks = len(graph_memory.tracks)
    num_events = len(graph_memory.events)
    total_track_frames = sum(len(track.frames) for track in graph_memory.tracks.values())
    avg_frames_per_track = (
        total_track_frames / num_tracks if num_tracks > 0 else 0.0
    )
    compression_ratio = (
        num_frames / num_events if num_events > 0 else float("inf")
    )

    print(f"Saved GraphMemory to {args.output}")
    print(f"Tracks: {num_tracks}")
    print(f"Events: {num_events}")
    print(f"Average frames per track: {avg_frames_per_track:.2f}")
    if compression_ratio == float("inf"):
        print("Compression ratio (frames/events): inf")
    else:
        print(f"Compression ratio (frames/events): {compression_ratio:.2f}")


if __name__ == "__main__":
    main()

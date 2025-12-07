"""
Benchmark harness comparing GraphMemory QA against video frame baselines.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

from graph_memory import Event, GraphMemory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GRAPH_MEMORY_JSON = Path("graph_memory.json")
QUESTIONS_JSON = Path("questions.json")
VIDEO_PATH = Path("video.mp4")

SHORT_WINDOW_SECONDS = 10.0
SHORT_WINDOW_SAMPLES = 8
FRAME_SAMPLING_COUNT = 16
GRAPH_TOP_K = 5

# Approximate memory per image fed to a VLM (in MB). Update once exact frame size is known.
ASSUMED_FRAME_SIZE_MB = 0.75


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_graph_memory(path: Path) -> GraphMemory:
    """Load GraphMemory JSON into the dataclass."""
    data = json.loads(path.read_text())
    return GraphMemory.from_dict(data)


def load_questions(path: Path) -> List[dict]:
    """Load ground-truth QA pairs."""
    with path.open("r") as f:
        return json.load(f)


def estimate_video_duration(graph: GraphMemory) -> float:
    """Estimate video duration using the last timestamp seen in tracks/events."""
    max_track_time = max((track.times[-1] for track in graph.tracks.values() if track.times), default=0.0)
    max_event_time = max((event.time_s for event in graph.events), default=0.0)
    return max(max_track_time, max_event_time)


def evaluate_answer(predicted: str, ground_truth: str) -> bool:
    """
    Simple placeholder scoring.
    TODO: replace with semantic match against ground-truth answers.
    """
    predicted_norm = predicted.strip().lower()
    ground_truth_norm = ground_truth.strip().lower()
    return predicted_norm in ground_truth_norm or ground_truth_norm in predicted_norm


def measure_graph_memory_size_mb(path: Path) -> float:
    """Return on-disk GraphMemory size in MB."""
    return path.stat().st_size / (1024 ** 2)


# ---------------------------------------------------------------------------
# Frame sampling placeholders
# ---------------------------------------------------------------------------
def sample_frames_short_window(video_path: Path, duration: float, num_samples: int, window_seconds: float) -> List[str]:
    """
    Placeholder for short-window frame extraction.
    TODO: Implement with OpenCV (cv2.VideoCapture) to pull frames from the end of the video.
    """
    if duration <= 0:
        times = [0.0 for _ in range(num_samples)]
    else:
        start_time = max(duration - window_seconds, 0.0)
        end_time = duration
        step = (end_time - start_time) / max(num_samples - 1, 1)
        times = [start_time + idx * step for idx in range(num_samples)]
    # TODO: Replace string descriptors with actual frame tensors or captions.
    return [f"frame_{idx}_at_{timestamp:.2f}s" for idx, timestamp in enumerate(times)]


def sample_frames_uniform(video_path: Path, duration: float, num_samples: int) -> List[str]:
    """
    Placeholder for uniform frame sampling across an entire video.
    TODO: Replace with actual frame extraction via OpenCV.
    """
    if duration <= 0:
        times = [0.0 for _ in range(num_samples)]
    else:
        step = duration / max(num_samples, 1)
        times = [min(idx * step, duration) for idx in range(num_samples)]
    return [f"frame_{idx}_at_{timestamp:.2f}s" for idx, timestamp in enumerate(times)]


# ---------------------------------------------------------------------------
# QA methods
# ---------------------------------------------------------------------------
def baseline_short_window_qa(question: dict, video_duration: float) -> Dict[str, object]:
    """
    Short-window baseline that only inspects frames near the end of the video.
    """
    start = time.perf_counter()
    frames = sample_frames_short_window(
        video_path=VIDEO_PATH,
        duration=video_duration,
        num_samples=SHORT_WINDOW_SAMPLES,
        window_seconds=SHORT_WINDOW_SECONDS,
    )
    prompt = (
        "You see frames from the end of a video. "
        f"Frames: {frames}. "
        f"Question: {question['q']} "
        "Short answer:"
    )
    # TODO: Replace with actual VLM call using sampled frames or their captions.
    answer = "TODO_short_window_answer"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    memory_mb = len(frames) * ASSUMED_FRAME_SIZE_MB
    correct = evaluate_answer(answer, question["a"])
    return {"answer": answer, "correct": correct, "time_ms": elapsed_ms, "memory_mb": memory_mb, "prompt": prompt}


def baseline_frame_sample_qa(question: dict, video_duration: float) -> Dict[str, object]:
    """
    Uniform frame sampling baseline for long videos.
    """
    start = time.perf_counter()
    frames = sample_frames_uniform(
        video_path=VIDEO_PATH,
        duration=video_duration,
        num_samples=FRAME_SAMPLING_COUNT,
    )
    prompt = (
        "You see frames sampled across the full video. "
        f"Frames: {frames}. "
        f"Question: {question['q']} "
        "Short answer:"
    )
    # TODO: Replace with actual VLM call taking sampled frames/captions as input.
    answer = "TODO_frame_sampling_answer"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    memory_mb = len(frames) * ASSUMED_FRAME_SIZE_MB
    correct = evaluate_answer(answer, question["a"])
    return {"answer": answer, "correct": correct, "time_ms": elapsed_ms, "memory_mb": memory_mb, "prompt": prompt}


def score_event_relevance(question_text: str, event: Event) -> float:
    """
    Lightweight lexical relevance score between a question and an event description.
    TODO: Replace with embedding similarity using a sentence-transformer or API.
    """
    question_words = set(question_text.lower().split())
    event_words = set(event.description.lower().split())
    overlap = question_words & event_words
    if not question_words:
        return 0.0
    return len(overlap) / len(question_words)


def retrieve_top_events(question_text: str, events: Iterable[Event], top_k: int) -> List[Event]:
    """Return the K events with highest lexical overlap to the question."""
    scored: List[Tuple[float, Event]] = []
    for event in events:
        score = score_event_relevance(question_text, event)
        scored.append((score, event))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [event for _, event in scored[:top_k]]


def graph_memory_qa(question: dict, graph: GraphMemory, graph_path: Path) -> Dict[str, object]:
    """
    GraphMemory QA that filters the events timeline before asking an LLM.
    """
    start = time.perf_counter()
    top_events = retrieve_top_events(question["q"], graph.events, top_k=GRAPH_TOP_K)
    if not top_events:
        event_lines = ["(no relevant events retrieved)"]
    else:
        event_lines = [f"{idx + 1}. [{event.time_s:.2f}s] {event.description}" for idx, event in enumerate(top_events)]
    prompt = "Video events timeline:\n" + "\n".join(event_lines) + f"\nQuestion: {question['q']}\nShort answer:"
    # TODO: Replace with LLM call conditioned on the retrieved timeline.
    answer = "TODO_graph_memory_answer"
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    memory_mb = measure_graph_memory_size_mb(graph_path)
    correct = evaluate_answer(answer, question["a"])
    return {"answer": answer, "correct": correct, "time_ms": elapsed_ms, "memory_mb": memory_mb, "prompt": prompt}


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------
def run_method(method: str, question: dict, graph: GraphMemory, graph_duration: float) -> Dict[str, object]:
    """Dispatch calls for each QA method."""
    if method == "short_window":
        return baseline_short_window_qa(question, video_duration=graph_duration)
    if method == "frame_sampling":
        return baseline_frame_sample_qa(question, video_duration=graph_duration)
    if method == "graph_memory":
        return graph_memory_qa(question, graph=graph, graph_path=GRAPH_MEMORY_JSON)
    raise ValueError(f"Unknown method: {method}")


def save_results_csv(results: List[dict], path: Path) -> None:
    """Save per-question results to CSV."""
    fieldnames = ["question", "method", "correct", "time_ms", "memory_mb", "answer"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "question": row["question"],
                "method": row["method"],
                "correct": row["correct"],
                "time_ms": f"{row['time_ms']:.2f}",
                "memory_mb": f"{row['memory_mb']:.2f}",
                "answer": row["answer"],
            })


def plot_accuracy(results: List[dict], output_path: Path) -> None:
    """Plot accuracy per method."""
    methods = sorted({row["method"] for row in results})
    accuracies = []
    for method in methods:
        rows = [row for row in results if row["method"] == method]
        if rows:
            accuracy = sum(1 for row in rows if row["correct"]) / len(rows)
        else:
            accuracy = 0.0
        accuracies.append(accuracy)

    plt.figure(figsize=(6, 4))
    plt.bar(methods, accuracies, color="skyblue")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Method")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_memory(results: List[dict], output_path: Path) -> None:
    """Plot memory footprint per method."""
    methods = sorted({row["method"] for row in results})
    memories = []
    for method in methods:
        rows = [row for row in results if row["method"] == method]
        if rows:
            mem = statistics.mean(row["memory_mb"] for row in rows)
        else:
            mem = 0.0
        memories.append(mem)

    plt.figure(figsize=(6, 4))
    plt.bar(methods, memories, color="salmon")
    plt.ylabel("Memory (MB)")
    plt.title("Memory vs Method")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_results(results: List[dict]) -> Dict[str, dict]:
    """Aggregate average metrics per method."""
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in results:
        grouped[row["method"]].append(row)

    summary: Dict[str, dict] = {}
    for method, rows in grouped.items():
        accuracy = sum(1 for row in rows if row["correct"]) / len(rows)
        avg_memory = statistics.mean(row["memory_mb"] for row in rows)
        avg_time = statistics.mean(row["time_ms"] for row in rows)
        summary[method] = {
            "accuracy": accuracy,
            "memory_mb": avg_memory,
            "time_ms": avg_time,
        }
    return summary


def print_summary(summary: Dict[str, dict]) -> None:
    """Print a formatted summary table."""
    print("Method Summary:")
    for method, stats in summary.items():
        acc = stats["accuracy"]
        mem = stats["memory_mb"]
        t = stats["time_ms"]
        print(
            f"{method:>12} | acc={acc:.2f} | mem={mem:.2f}MB | avg_time={t:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    graph = load_graph_memory(GRAPH_MEMORY_JSON)
    questions = load_questions(QUESTIONS_JSON)
    video_duration = estimate_video_duration(graph)

    methods = ["short_window", "frame_sampling", "graph_memory"]
    results: List[dict] = []

    for question in questions:
        for method in methods:
            out = run_method(method, question, graph=graph, graph_duration=video_duration)
            results.append(
                {
                    "question": question["q"],
                    "method": method,
                    "answer": out["answer"],
                    "correct": out["correct"],
                    "time_ms": out["time_ms"],
                    "memory_mb": out["memory_mb"],
                }
            )

    results_path = Path("results.csv")
    save_results_csv(results, results_path)
    plot_accuracy(results, output_path=Path("accuracy_vs_method.png"))
    plot_memory(results, output_path=Path("memory_vs_method.png"))

    summary = summarize_results(results)
    print_summary(summary)
    print(f"\nSaved raw results to {results_path}")


if __name__ == "__main__":
    main()

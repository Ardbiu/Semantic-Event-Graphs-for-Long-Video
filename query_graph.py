import json
from pathlib import Path


def load_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_ts(seconds: float) -> str:
    if seconds is None:
        return "00:00"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def generate_narrative(events):
    lines = []
    for evt in events:
        ts = format_ts(evt.get("timestamp", 0.0))
        subj = evt.get("subject", "subject")
        obj = evt.get("object", "object")
        if evt.get("type") == "START":
            verb = "picked up"
        elif evt.get("type") == "END":
            verb = "put down"
        else:
            verb = "interacted with"
        lines.append(f"[{ts}] {subj} {verb} {obj}.")
    return "\n".join(lines)


def count_words(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def build_prompt(narrative: str, user_question: str) -> str:
    system_prompt = (
        "You are a helpful video assistant. "
        "Here is a log of events from a video. "
        "Answer the user's question based ONLY on this log."
    )
    return (
        f"System:\n{system_prompt}\n\n"
        f"Narrative:\n{narrative}\n\n"
        f"User Question:\n{user_question}\n"
    )


def main():
    event_path = Path("event_log.json")
    if not event_path.exists():
        raise SystemExit(f"Missing {event_path}. Run the tracker first.")

    events = load_events(event_path)
    narrative = generate_narrative(events)
    word_count = count_words(narrative)
    print(f"Compressed Context Size: {word_count} words vs Estimated Video Size: Millions of pixels.")
    print("\n--- Narrative ---")
    print(narrative if narrative else "(No events)")
    print("-----------------\n")

    while True:
        try:
            question = input("Enter a question about the video (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if question.lower() in {"exit", "quit", ""}:
            print("Exiting.")
            break
        final_prompt = build_prompt(narrative, question)
        print("\n--- FINAL PROMPT (copy/paste to ChatGPT) ---")
        print(final_prompt)
        print("-------------------------------------------\n")


if __name__ == "__main__":
    main()

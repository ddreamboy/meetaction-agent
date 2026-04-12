import argparse
import json
import sys
import time
from pathlib import Path

import httpx

TRANSCRIPTS_DIR = Path(__file__).parent / "synthetic_transcripts"


def run_transcript(api_base: str, name: str, transcript: str) -> dict:
    start = time.monotonic()
    try:
        resp = httpx.post(
            f"{api_base}/process_text",
            json={"transcript": transcript},
            timeout=300,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        if resp.status_code != 200:
            return {
                "name": name,
                "status": "error",
                "error": resp.text,
                "latency_ms": latency_ms,
            }

        data = resp.json()
        return {
            "name": name,
            "status": data.get("status", "unknown"),
            "session_id": data.get("session_id"),
            "proposed_output": data.get("proposed_output", []),
            "error_message": data.get("error_message"),
            "progress_steps": data.get("progress_steps", []),
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "exception",
            "error": str(exc),
            "latency_ms": int((time.monotonic() - start) * 1000),
        }


def format_markdown(results: list[dict]) -> str:
    lines = ["# Eval Results\n"]
    ok = sum(1 for r in results if r["status"] == "awaiting_confirmation")
    lines.append(f"**Итого:** {len(results)} транскриптов, {ok} успешно\n")

    for r in results:
        lines.append(f"---\n## {r['name']}")
        lines.append(f"- **Статус:** {r['status']}  |  **Время:** {r['latency_ms']} ms")

        if r.get("error"):
            lines.append(f"- **Ошибка:** {r['error']}")

        if r.get("error_message"):
            lines.append(f"- **Ошибка агента:** {r['error_message']}")

        steps = r.get("progress_steps", [])
        if steps:
            lines.append(f"- **Шаги:** {' -> '.join(steps)}")

        proposed = r.get("proposed_output", [])
        if proposed:
            lines.append(f"\n**Предложено ({len(proposed)} шт.):**")
            for item in proposed:
                if "title" in item:
                    assignees = item.get("assignees") or []
                    due = item.get("due_string") or ""
                    meta = []
                    if assignees:
                        meta.append(f"отв: {', '.join(assignees)}")
                    if due:
                        meta.append(f"срок: {due}")
                    meta_str = f" [{', '.join(meta)}]" if meta else ""
                    lines.append(f"  - {item['title']}{meta_str}")
                    if item.get("description"):
                        lines.append(f"    > {item['description']}")
                elif "topic" in item:
                    lines.append(f"  - **{item['topic']}**")
                    for kp in item.get("key_points", []):
                        lines.append(f"    - {kp}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run evals on synthetic transcripts")
    parser.add_argument("--api-base", default="http://localhost:7007")
    parser.add_argument("--out", default="evals_result.md")
    parser.add_argument(
        "--json", action="store_true", help="Дополнительно сохранить JSON"
    )
    args = parser.parse_args()

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not transcript_files:
        print(f"Транскрипты не найдены в {TRANSCRIPTS_DIR}", file=sys.stderr)
        sys.exit(1)

    try:
        httpx.get(f"{args.api_base}/health", timeout=5).raise_for_status()
    except Exception as exc:
        print(f"API недоступен: {args.api_base} — {exc}", file=sys.stderr)
        sys.exit(1)

    results = []
    for path in transcript_files:
        transcript = path.read_text(encoding="utf-8").strip()
        print(f"[{path.name}] обрабатываю...", end=" ", flush=True)
        result = run_transcript(args.api_base, path.name, transcript)
        results.append(result)
        status_emoji = "[+]" if result["status"] == "awaiting_confirmation" else "[x]"
        print(f"{status_emoji} {result['latency_ms']}ms")

    md = format_markdown(results)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nРезультаты сохранены: {args.out}")

    if args.json:
        json_path = Path(args.out).with_suffix(".json")
        json_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON сохранён: {json_path}")


if __name__ == "__main__":
    main()

import json
import os
import re

import gradio as gr
import httpx

API_BASE = os.getenv("API_BASE_URL", "http://localhost:7007")
UI_PORT = int(os.getenv("UI_PORT", "7860"))

_session_state: dict = {"session_id": None}


def _extract_speakers(transcript: str) -> list[str]:
    seen: set[str] = set()
    result = []
    for m in re.finditer(r"(?:^|\]\s*)(speaker_\d+):", transcript, re.MULTILINE):
        sp = m.group(1)
        if sp not in seen:
            seen.add(sp)
            result.append(sp)
    return sorted(result, key=lambda s: int(s.split("_")[1]))


def _render_status(data: dict) -> str:
    lines = [
        f"Обработка завершена. Session: {data.get('session_id', '-')}",
        f"Статус: {data.get('status', '-')}",
    ]
    progress_steps = data.get("progress_steps", [])
    if progress_steps:
        lines.append("Шаги:")
        for idx, step in enumerate(progress_steps, start=1):
            lines.append(f"{idx}. {step}")
    current_step = data.get("current_step")
    if current_step:
        lines.append(f"Текущий шаг: {current_step}")
    if data.get("error_message"):
        lines.append(f"Ошибка: {data['error_message']}")
    return "\n".join(lines)


# --- File processing ---


def _process_response(data: dict, response_status: int):
    if response_status != 200:
        hidden = gr.update(visible=False)
        return (
            f"Ошибка: {data}",
            "",
            "",
            hidden,
            hidden,
            hidden,
            [],
            "",
            [],
        )

    _session_state["session_id"] = data["session_id"]
    proposed = data.get("proposed_output", [])
    proposed_text = json.dumps(proposed, ensure_ascii=False, indent=2)
    transcript_text = data.get("transcript_text", "")
    can_confirm = data.get("status") == "awaiting_confirmation"

    speakers = _extract_speakers(transcript_text)
    speaker_table = [[sp, sp] for sp in speakers]

    initial_chat: list[dict] = []
    if can_confirm:
        tasks_count = len(proposed)
        word_count = len(transcript_text.split())
        initial_chat = [
            {
                "role": "user",
                "content": (
                    f"[Транскрипция добавлена: {word_count} слов]\n\n"
                    f"Выделенные задачи ({tasks_count} шт.):\n```json\n{proposed_text}\n```"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Контекст встречи загружен. "
                    "Если нужно что-то скорректировать — напишите, я учту правки и обновлю список."
                ),
            },
        ]

    return (
        _render_status(data),
        proposed_text,
        transcript_text,
        gr.update(visible=can_confirm),
        gr.update(visible=bool(speakers)),
        gr.update(visible=can_confirm),
        initial_chat,
        "",
        speaker_table,
    )


def upload_and_process(file):
    hidden = gr.update(visible=False)
    if file is None:
        return "Файл не выбран", "", "", hidden, hidden, hidden, [], "", []

    with open(file.name, "rb") as f:
        response = httpx.post(
            f"{API_BASE}/process",
            files={"file": (file.name, f)},
            timeout=600,
        )

    data = response.json() if response.status_code == 200 else response.text
    return _process_response(data, response.status_code)


def process_text_input(transcript: str):
    hidden = gr.update(visible=False)
    transcript = (transcript or "").strip()
    if not transcript:
        return "Введите текст транскрипции", "", "", hidden, hidden, hidden, [], "", []

    response = httpx.post(
        f"{API_BASE}/process_text",
        json={"transcript": transcript},
        timeout=300,
    )

    data = response.json() if response.status_code == 200 else response.text
    return _process_response(data, response.status_code)


def rename_speakers(speaker_table, chat_history: list[dict]):
    session_id = _session_state["session_id"]
    if not session_id:
        return "Нет активной сессии", gr.update(), gr.update(), chat_history

    if hasattr(speaker_table, "values"):
        rows = speaker_table.values.tolist()
    else:
        rows = speaker_table or []

    speaker_map = {
        str(row[0]): str(row[1])
        for row in rows
        if row[0] and row[1] and str(row[0]) != str(row[1])
    }
    if not speaker_map:
        return "Нет изменений для применения", gr.update(), gr.update(), chat_history

    response = httpx.post(
        f"{API_BASE}/rename_speakers",
        json={"session_id": session_id, "speaker_map": speaker_map},
        timeout=30,
    )
    if response.status_code != 200:
        return f"Ошибка: {response.text}", gr.update(), gr.update(), chat_history

    data = response.json()
    proposed_text = json.dumps(data["proposed_output"], ensure_ascii=False, indent=2)

    # Формируем сообщение об изменениях
    rename_lines = [f"{old} -> {new}" for old, new in speaker_map.items()]
    rename_summary = "\n".join(rename_lines)
    chat_msg = {
        "role": "assistant",
        "content": (
            f"Имена спикеров обновлены:\n{rename_summary}\n\n"
            "Задачи и транскрипция обновлены с новыми именами."
        ),
    }
    updated_chat = chat_history + [chat_msg]

    return "Спикеры переименованы", proposed_text, data["transcript_text"], updated_chat


# --- Asign tasks ---


def confirm_tasks():
    if not _session_state["session_id"]:
        return "Нет активной сессии"

    response = httpx.post(
        f"{API_BASE}/confirm",
        json={"session_id": _session_state["session_id"]},
        timeout=120,
    )
    if response.status_code != 200:
        return f"Ошибка: {response.text}"

    data = response.json()
    task_ids = data.get("created_task_ids", [])
    error = data.get("error_message")
    grouping = data.get("task_grouping", {})

    lines = [f"Подтверждено. Создано задач в Todoist: {len(task_ids)}"]

    by_assignee = grouping.get("by_assignee", {})
    if by_assignee:
        lines.append("\nЗадачи по одному ответственному:")
        for assignee, tasks in by_assignee.items():
            tasks_str = "; ".join(tasks) if len(tasks) > 1 else tasks[0]
            merged = " (объединены)" if len(tasks) > 1 else ""
            lines.append(f"  {assignee}{merged}: {tasks_str}")

    multiple = grouping.get("multiple_assignees", [])
    if multiple:
        lines.append("\nЗадачи с несколькими ответственными:")
        for item in multiple:
            assignees_str = ", ".join(item["assignees"])
            lines.append(f"  [{assignees_str}]: {item['title']}")

    no_assignee = grouping.get("no_assignee", [])
    if no_assignee:
        lines.append("\nЗадачи без ответственного:")
        for t in no_assignee:
            lines.append(f"  {t}")

    if error:
        lines.append(f"\nПредупреждение: {error}")

    return "\n".join(lines)


# --- Chat ---


def send_chat_message(
    user_message: str, chat_history: list[dict], bot_understanding: str
):
    if not user_message.strip():
        return chat_history, bot_understanding, ""

    session_id = _session_state["session_id"]
    if not session_id:
        return (
            chat_history
            + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "Нет активной сессии."},
            ],
            bot_understanding,
            "",
        )

    try:
        response = httpx.post(
            f"{API_BASE}/clarify",
            json={
                "session_id": session_id,
                "message": user_message,
                "chat_history": chat_history,
            },
            timeout=60,
        )
        bot_msg = (
            response.json()["bot_message"]
            if response.status_code == 200
            else f"Ошибка: {response.text}"
        )
    except Exception as exc:
        bot_msg = f"Ошибка соединения: {exc}"

    new_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": bot_msg},
    ]
    return new_history, bot_msg, ""


def apply_changes(chat_history: list, bot_understanding: str):
    session_id = _session_state["session_id"]
    if not session_id:
        return "Нет активной сессии", gr.update(), chat_history, ""

    user_messages = [
        msg["content"]
        for msg in chat_history
        if msg["role"] == "user"
        and not msg["content"].startswith("[Транскрипция добавлена")
    ]
    if not user_messages:
        return "Сначала обсудите правки в чате", gr.update(), chat_history, ""

    feedback_parts = ["Запросы пользователя:"] + user_messages
    if bot_understanding.strip():
        feedback_parts += ["\nПонимание ассистента:", bot_understanding.strip()]
    feedback = "\n".join(feedback_parts)

    response = httpx.post(
        f"{API_BASE}/refine",
        json={"session_id": session_id, "feedback": feedback},
        timeout=120,
    )
    if response.status_code != 200:
        return f"Ошибка: {response.text}", gr.update(), chat_history, ""

    data = response.json()
    proposed = data.get("proposed_output", [])
    proposed_text = json.dumps(proposed, ensure_ascii=False, indent=2)
    tasks_count = len(proposed)
    updated_chat = chat_history + [
        {
            "role": "assistant",
            "content": (
                f"Правки применены. Список задач обновлён ({tasks_count} шт.) — "
                "смотрите блок «Предложенные задачи / инсайты» выше."
            ),
        }
    ]
    return _render_status(data), proposed_text, updated_chat, ""


# --- RAG ---


def rag_query(query: str = "", meeting_type: str = "Все"):
    query = (query or "").strip()
    if not query:
        return "Введите вопрос", ""

    payload: dict = {"query": query}
    if meeting_type and meeting_type != "Все":
        payload["meeting_type"] = (
            "work_meeting" if meeting_type == "Рабочие" else "consultation"
        )

    response = httpx.post(f"{API_BASE}/query", json=payload, timeout=60)
    if response.status_code != 200:
        return f"Ошибка: {response.text}", ""

    data = response.json()
    answer = f"[Источников: {data['sources_count']}]\n\n{data['answer']}"

    sources = data.get("sources", [])
    if sources:
        parts = []
        for i, s in enumerate(sources, 1):
            mt = "Рабочая" if s.get("meeting_type") == "work_meeting" else "Консультация"
            parts.append(
                f"[{i}] {s.get('date', '?')} | {mt} | {s.get('participants_count', 0)} уч.\n"
                f"{s.get('summary', '').strip()}"
            )
        sources_text = "\n\n".join(parts)
    else:
        sources_text = ""

    return answer, sources_text


# --- UI ---

with gr.Blocks(title="MeetAction Agent") as demo:
    gr.Markdown("# MeetAction Agent\nОбработка записей рабочих встреч")

    bot_understanding_state = gr.State(value="")
    chat_history_state = gr.State(value=[])

    with gr.Tab("Обработка встречи"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Аудио/видео файл",
                    file_types=[".mp3", ".mp4", ".wav", ".m4a", ".webm"],
                )
                process_btn = gr.Button("Загрузить и обработать", variant="primary")
            with gr.Column():
                transcript_input = gr.Textbox(
                    label="Или вставить транскрипцию текстом",
                    placeholder="speaker_0: Добрый день...\nspeaker_1: Привет...",
                    lines=5,
                )
                process_text_btn = gr.Button("Обработать транскрипцию", variant="primary")

        status_out = gr.Textbox(label="Статус", lines=10)

        with gr.Row():
            with gr.Column():
                proposed_out = gr.Textbox(
                    label="Предложенные задачи / инсайты", lines=14
                )
                copy_tasks_btn = gr.Button("Копировать задачи", size="sm")
            with gr.Column():
                transcript_out = gr.Textbox(label="Транскрипция", lines=14)
                copy_transcript_btn = gr.Button("Копировать транскрипцию", size="sm")

        with gr.Column(visible=False) as speaker_row:
            gr.Markdown("#### Спикеры - задайте имена (одинаковое имя = один человек)")
            speaker_table = gr.Dataframe(
                headers=["Спикер", "Имя"],
                datatype=["str", "str"],
                column_count=(2, "fixed"),
                row_count=(1, "dynamic"),
                interactive=True,
                label="",
            )
            rename_btn = gr.Button("Применить имена", size="sm")

        result_out = gr.Textbox(label="Результат", lines=3)

        with gr.Column(visible=False) as chat_row:
            gr.Markdown("### Чат для уточнения задач")
            chatbot = gr.Chatbot(label="Обсуждение правок", height=320)
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Опишите что нужно изменить...",
                    label="",
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Отправить", scale=1)
            apply_btn = gr.Button("Применить правки", variant="secondary")

        with gr.Row(visible=False) as action_row:
            confirm_btn = gr.Button(
                "Подтвердить и создать задачи в Todoist", variant="primary"
            )

        _processing_outputs = [
            status_out,
            proposed_out,
            transcript_out,
            action_row,
            speaker_row,
            chat_row,
            chat_history_state,
            bot_understanding_state,
            speaker_table,
        ]

        process_btn.click(
            upload_and_process,
            inputs=[file_input],
            outputs=_processing_outputs,
        ).then(lambda h: h, inputs=[chat_history_state], outputs=[chatbot])

        process_text_btn.click(
            process_text_input,
            inputs=[transcript_input],
            outputs=_processing_outputs,
        ).then(lambda h: h, inputs=[chat_history_state], outputs=[chatbot])

        rename_btn.click(
            rename_speakers,
            inputs=[speaker_table, chat_history_state],
            outputs=[result_out, proposed_out, transcript_out, chat_history_state],
        ).then(lambda h: h, inputs=[chat_history_state], outputs=[chatbot])

        confirm_btn.click(confirm_tasks, outputs=[result_out])

        copy_tasks_btn.click(
            None,
            inputs=[proposed_out],
            outputs=[],
            js="(t) => { navigator.clipboard.writeText(t); }",
        )
        copy_transcript_btn.click(
            None,
            inputs=[transcript_out],
            outputs=[],
            js="(t) => { navigator.clipboard.writeText(t); }",
        )

        def _send(msg, history, understanding):
            return send_chat_message(msg, history, understanding)

        send_btn.click(
            _send,
            inputs=[chat_input, chat_history_state, bot_understanding_state],
            outputs=[chat_history_state, bot_understanding_state, chat_input],
        ).then(lambda h: h, inputs=[chat_history_state], outputs=[chatbot])

        chat_input.submit(
            _send,
            inputs=[chat_input, chat_history_state, bot_understanding_state],
            outputs=[chat_history_state, bot_understanding_state, chat_input],
        ).then(lambda h: h, inputs=[chat_history_state], outputs=[chatbot])

        apply_btn.click(
            apply_changes,
            inputs=[chat_history_state, bot_understanding_state],
            outputs=[
                result_out,
                proposed_out,
                chat_history_state,
                bot_understanding_state,
            ],
        ).then(lambda h: h, inputs=[chat_history_state], outputs=[chatbot])

    with gr.Tab("Вопросы по встречам"):
        query_in = gr.Textbox(label="Вопрос", placeholder="Кто обещал сдать отчет?")
        meeting_type_filter = gr.Radio(
            choices=["Все", "Рабочие", "Консультации"],
            value="Все",
            label="Тип встречи",
        )
        query_btn = gr.Button("Спросить", variant="primary")
        answer_out = gr.Textbox(label="Ответ", lines=6)
        sources_out = gr.Textbox(label="Источники", lines=8, visible=False)

        def _rag_query(q, mt):
            answer, sources = rag_query(q, mt)
            return answer, gr.update(value=sources, visible=bool(sources))

        query_btn.click(
            _rag_query,
            inputs=[query_in, meeting_type_filter],
            outputs=[answer_out, sources_out],
        )
        query_in.submit(
            _rag_query,
            inputs=[query_in, meeting_type_filter],
            outputs=[answer_out, sources_out],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=UI_PORT)

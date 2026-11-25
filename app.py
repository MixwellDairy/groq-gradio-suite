import base64
import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHATS_FILE = "chats.json"
UI_FILE = Path(__file__).with_name("index.html")
MAX_REQUEST_SIZE = 35 * 1024 * 1024
MAX_VOICE_HISTORY_MESSAGES = 16
chats_lock = threading.Lock()
client_lock = threading.Lock()
client = None


def memory_limit_bytes():
    limits = []
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            value = path.read_text(encoding="ascii").strip()
            if value != "max":
                limit = int(value)
                if 0 < limit < 1 << 60:
                    limits.append(limit)
        except (OSError, ValueError):
            pass
    if limits:
        return min(limits)
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None


def request_worker_limit():
    configured_limit = os.environ.get("MAX_REQUEST_WORKERS")
    if configured_limit:
        try:
            return max(1, int(configured_limit))
        except ValueError:
            pass

    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 1
    cpu_limit = max(2, cpu_count * 4)
    memory_limit = memory_limit_bytes()
    if memory_limit is None:
        return min(cpu_limit, 32)
    memory_workers = max(2, memory_limit // (256 * 1024 * 1024))
    return min(cpu_limit, memory_workers)


def get_client():
    global client
    if client is None:
        with client_lock:
            if client is None:
                from dotenv import load_dotenv
                from groq import Groq

                load_dotenv()
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return client


def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, dict) and data:
                    return data
        except (OSError, ValueError) as error:
            print(f"Error loading chats: {error}")
    chat_id = str(uuid.uuid4())
    return {chat_id: {"title": "New Chat", "messages": []}}


def save_chats(chats):
    try:
        with open(CHATS_FILE, "w", encoding="utf-8") as file:
            json.dump(chats, file, separators=(",", ":"))
    except OSError as error:
        print(f"Error saving chats: {error}")


def search_web(query):
    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=3))
        if results:
            return "\n\n".join(
                f"Source: {result.get('title')} ({result.get('href')})\n{result.get('body')}"
                for result in results
            )
    except Exception as error:
        print(f"Search error: {error}")
    return ""


def audio_bytes_to_base64(response):
    return base64.b64encode(response.read()).decode("ascii")


def transcribe_audio(audio_data, filename, model_name):
    return get_client().audio.transcriptions.create(
        file=(os.path.basename(filename), audio_data),
        model=model_name,
        response_format="text",
    )


def generate_speech(text, model_name, voice_name):
    response = get_client().audio.speech.create(
        model=model_name,
        voice=voice_name.lower(),
        input=text,
        response_format="wav",
    )
    return audio_bytes_to_base64(response)


def sanitize_voice_history(history):
    if not isinstance(history, list):
        return []
    sanitized = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            sanitized.append({"role": role, "content": content})
    return sanitized[-MAX_VOICE_HISTORY_MESSAGES:]


def run_voice_agent(audio_data, filename, agent_goal, agent_history):
    user_speech_text = transcribe_audio(
        audio_data, filename, "whisper-large-v3-turbo"
    )
    if not user_speech_text or not user_speech_text.strip():
        return "(No speech detected)", "", None, sanitize_voice_history(agent_history)

    custom_goal = agent_goal.strip() if isinstance(agent_goal, str) else ""
    system_instructions = custom_goal or (
        "You are a helpful, conversational voice assistant."
    )
    system_instructions += (
        "\n\nFor this spoken conversation: keep replies concise, use natural "
        "conversational language, avoid markdown tables and long lists, and "
        "do not repeat the user's words unless useful."
    )
    messages = [{"role": "system", "content": system_instructions}]
    messages.extend(sanitize_voice_history(agent_history))
    messages.append({"role": "user", "content": user_speech_text.strip()})

    completion = get_client().chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.7,
        max_tokens=300,
    )
    agent_reply_text = (completion.choices[0].message.content or "").strip()
    if not agent_reply_text:
        agent_reply_text = "I'm sorry, I couldn't generate a response. Please try again."
    updated_history = sanitize_voice_history(agent_history) + [
        {"role": "user", "content": user_speech_text.strip()},
        {"role": "assistant", "content": agent_reply_text},
    ]
    updated_history = updated_history[-MAX_VOICE_HISTORY_MESSAGES:]
    response = get_client().audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="troy",
        input=agent_reply_text,
        response_format="wav",
    )
    return (
        user_speech_text.strip(),
        agent_reply_text,
        audio_bytes_to_base64(response),
        updated_history,
    )


class AppHandler(BaseHTTPRequestHandler):
    server_version = "GroqSuite/1.0"

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid Content-Length")
        if content_length < 0 or content_length > MAX_REQUEST_SIZE:
            raise ValueError("Request body is too large")
        body = self.rfile.read(content_length)
        data = json.loads(body or b"{}")
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def _send_audio(self, audio_data):
        self._send_json({"audio": audio_data})

    def _authorized_origin(self):
        origin = self.headers.get("Origin")
        if not origin:
            return True
        from urllib.parse import urlsplit

        parsed_origin = urlsplit(origin)
        return (
            parsed_origin.scheme in ("http", "https")
            and parsed_origin.netloc == self.headers.get("Host")
        )

    def do_GET(self):
        if self.path == "/":
            try:
                body = UI_FILE.read_bytes()
            except OSError:
                self.send_error(500, "UI file is unavailable")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/chats":
            with chats_lock:
                self._send_json(load_chats())
        else:
            self.send_error(404)

    def do_POST(self):
        if not self._authorized_origin():
            self._send_json({"error": "Invalid request origin"}, 403)
            return
        try:
            data = self._read_json()
            if self.path == "/api/chats":
                with chats_lock:
                    chats = load_chats()
                    chat_id = str(uuid.uuid4())
                    chats[chat_id] = {"title": "New Chat", "messages": []}
                    save_chats(chats)
                self._send_json({"id": chat_id, "chats": chats})
            elif self.path == "/api/chat":
                self._stream_chat(data)
            elif self.path == "/api/stt":
                audio_data = base64.b64decode(data["audio"], validate=True)
                text = transcribe_audio(
                    audio_data,
                    data.get("filename", "recording.webm"),
                    data["model"],
                )
                self._send_json({"text": text})
            elif self.path == "/api/tts":
                text = data.get("text", "")
                if not text.strip():
                    self._send_json({"error": "Please enter some text first."}, 400)
                    return
                audio = generate_speech(text, data["model"], data["voice"])
                self._send_audio(audio)
            elif self.path == "/api/voice":
                audio_data = base64.b64decode(data["audio"], validate=True)
                speech, reply, audio, history = run_voice_agent(
                    audio_data,
                    data.get("filename", "recording.webm"),
                    data.get("goal", ""),
                    data.get("history", []),
                )
                self._send_json(
                    {
                        "speech": speech,
                        "reply": reply,
                        "audio": audio,
                        "history": history,
                    }
                )
            else:
                self.send_error(404)
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)
        except Exception as error:
            self._send_json({"error": str(error)}, 500)

    def do_PUT(self):
        if not self._authorized_origin():
            self._send_json({"error": "Invalid request origin"}, 403)
            return
        try:
            data = self._read_json()
            chat_id = self.path.removeprefix("/api/chats/")
            if not chat_id or "/" in chat_id:
                self.send_error(404)
                return
            with chats_lock:
                chats = load_chats()
                if chat_id not in chats:
                    self._send_json({"error": "Conversation not found"}, 404)
                    return
                title = data.get("title", "").strip()
                if title:
                    chats[chat_id]["title"] = title
                    save_chats(chats)
            self._send_json({"chats": chats})
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, 400)

    def do_DELETE(self):
        if not self._authorized_origin():
            self._send_json({"error": "Invalid request origin"}, 403)
            return
        chat_id = self.path.removeprefix("/api/chats/")
        if not chat_id or "/" in chat_id:
            self.send_error(404)
            return
        with chats_lock:
            chats = load_chats()
            chats.pop(chat_id, None)
            if not chats:
                new_id = str(uuid.uuid4())
                chats[new_id] = {"title": "New Chat", "messages": []}
            save_chats(chats)
        selected_id = next(iter(chats))
        self._send_json({"id": selected_id, "chats": chats})

    def _stream_chat(self, data):
        message = data.get("message", "")
        chat_id = data.get("chat_id")
        if not isinstance(message, str) or not message.strip() or not chat_id:
            self._send_json({"error": "A message and conversation are required."}, 400)
            return

        with chats_lock:
            chats = load_chats()
            if chat_id not in chats:
                chats[chat_id] = {"title": "New Chat", "messages": []}
            chat = chats[chat_id]
            if not chat["messages"] and chat["title"] == "New Chat":
                trimmed = message.strip()
                chat["title"] = trimmed[:32] + ("..." if len(trimmed) > 32 else "")
            chat["messages"].append({"role": "user", "content": message})
            chat["messages"].append({"role": "assistant", "content": ""})
            save_chats(chats)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self._send_event(
            {
                "chat_id": chat_id,
                "chats": chats,
                "messages": chat["messages"],
            }
        )

        system_content = data.get("system_prompt", "").strip()
        if not system_content:
            system_content = "You are a helpful, laid-back AI assistant."
        if data.get("enable_search"):
            search_results = search_web(message)
            if search_results:
                system_content += (
                    "\n\nHere are some live web search results for context:\n"
                    + search_results
                )

        api_messages = [{"role": "system", "content": system_content}]
        api_messages.extend(chat["messages"][:-2])
        api_messages.append({"role": "user", "content": message})
        response_parts = []
        try:
            stream = get_client().chat.completions.create(
                model=data.get("model", "openai/gpt-oss-120b"),
                messages=api_messages,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    delta = chunk.choices[0].delta.content
                    response_parts.append(delta)
                    self._send_event(
                        {
                            "chat_id": chat_id,
                            "delta": delta,
                        }
                    )
            chat["messages"][-1]["content"] = "".join(response_parts)
        except Exception as error:
            chat["messages"][-1]["content"] = f"⚠️ **API Error:** {error}"

        with chats_lock:
            current_chats = load_chats()
            current_chats[chat_id] = chat
            save_chats(current_chats)
        self._send_event(
            {
                "chat_id": chat_id,
                "chats": current_chats,
                "messages": chat["messages"],
                "done": True,
            }
        )

    def _send_event(self, data):
        payload = json.dumps(data, ensure_ascii=False)
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")


class ResourceAwareHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler):
        self.request_limit = request_worker_limit()
        self.request_slots = threading.BoundedSemaphore(self.request_limit)
        super().__init__(server_address, request_handler)

    def process_request(self, request, client_address):
        if not self.request_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Retry-After: 1\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.request_slots.release()


if __name__ == "__main__":
    server = ResourceAwareHTTPServer(
        ("0.0.0.0", int(os.environ.get("PORT", 7860))), AppHandler
    )
    print(
        f"Groq Multimodal Suite listening on port {server.server_port} "
        f"with up to {server.request_limit} concurrent requests"
    )
    server.serve_forever()

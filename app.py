import base64
import json
import os
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHATS_FILE = "chats.json"
UI_FILE = Path(__file__).with_name("index.html")
MAX_REQUEST_SIZE = 35 * 1024 * 1024
chats_lock = threading.Lock()
client_lock = threading.Lock()
client = None


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
            json.dump(chats, file, indent=2)
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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        path = audio_file.name
    try:
        response.write_to_file(path)
        with open(path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("ascii")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


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


def run_voice_agent(audio_data, filename, agent_goal, agent_history):
    user_speech_text = transcribe_audio(
        audio_data, filename, "whisper-large-v3-turbo"
    )
    if not user_speech_text or not user_speech_text.strip():
        return "(No speech detected)", "", None, agent_history

    system_instructions = (
        agent_goal.strip()
        if agent_goal and agent_goal.strip()
        else "You are a helpful, conversational voice assistant. Keep your replies concise and natural for speech."
    )
    messages = [{"role": "system", "content": system_instructions}]
    messages.extend(agent_history)
    messages.append({"role": "user", "content": user_speech_text.strip()})

    completion = get_client().chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
    )
    agent_reply_text = completion.choices[0].message.content
    updated_history = list(agent_history) + [
        {"role": "user", "content": user_speech_text.strip()},
        {"role": "assistant", "content": agent_reply_text},
    ]
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
        partial_response = ""
        try:
            stream = get_client().chat.completions.create(
                model=data.get("model", "openai/gpt-oss-120b"),
                messages=api_messages,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    partial_response += chunk.choices[0].delta.content
                    chat["messages"][-1]["content"] = partial_response
                    self._send_event(
                        {
                            "chat_id": chat_id,
                            "delta": chunk.choices[0].delta.content,
                        }
                    )
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


if __name__ == "__main__":
    server = ThreadingHTTPServer(
        ("0.0.0.0", int(os.environ.get("PORT", 7860))), AppHandler
    )
    print(f"Groq Multimodal Suite listening on port {server.server_port}")
    server.serve_forever()

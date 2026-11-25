import os
import json
import uuid
import tempfile
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
from ddgs import DDGS

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CHATS_FILE = "chats.json"


def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
        except Exception as e:
            print(f"Error loading chats: {e}")
    default_id = str(uuid.uuid4())
    return {default_id: {"title": "New Chat", "messages": []}}


def save_chats(chats):
    try:
        with open(CHATS_FILE, "w", encoding="utf-8") as f:
            json.dump(chats, f, indent=2)
    except Exception as e:
        print(f"Error saving chats: {e}")


def get_dropdown_choices(chats):
    return [(data["title"], cid) for cid, data in chats.items()]


def search_web(query):
    try:
        results = list(DDGS().text(query, max_results=3))
        if results:
            return "\n\n".join(
                [
                    f"Source: {r.get('title')} ({r.get('href')})\n{r.get('body')}"
                    for r in results
                ]
            )
    except Exception as e:
        print(f"Search error: {e}")
    return ""


def respond(message, current_chat_id, custom_system_prompt, selected_model, enable_search):
    if not message or not message.strip():
        chats = load_chats()
        return "", chats.get(current_chat_id, {}).get("messages", []), gr.update()

    chats = load_chats()
    if current_chat_id not in chats:
        chats[current_chat_id] = {"title": "New Chat", "messages": []}

    chat_data = chats[current_chat_id]

    if len(chat_data["messages"]) == 0 and chat_data["title"] == "New Chat":
        trimmed = message.strip()
        chat_data["title"] = trimmed[:32] + ("..." if len(trimmed) > 32 else "")

    chat_data["messages"].append({"role": "user", "content": message})
    save_chats(chats)

    system_content = (
        custom_system_prompt.strip()
        if custom_system_prompt and custom_system_prompt.strip()
        else "You are a helpful, laid-back AI assistant."
    )
    if enable_search:
        search_results = search_web(message)
        if search_results:
            system_content += f"\n\nHere are some live web search results for context:\n{search_results}"

    api_messages = [{"role": "system", "content": system_content}]
    for msg in chat_data["messages"][:-1]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": message})

    chat_data["messages"].append({"role": "assistant", "content": ""})
    yield "", chat_data["messages"], gr.update(
        choices=get_dropdown_choices(chats), value=current_chat_id
    )

    try:
        stream = client.chat.completions.create(
            model=selected_model,
            messages=api_messages,
            stream=True,
        )

        partial_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                partial_response += chunk.choices[0].delta.content
                chat_data["messages"][-1]["content"] = partial_response
                yield "", chat_data["messages"], gr.update()

    except Exception as e:
        chat_data["messages"][-1]["content"] = f"⚠️ **API Error:** {str(e)}"

    chats[current_chat_id] = chat_data
    save_chats(chats)

    yield "", chat_data["messages"], gr.update(
        choices=get_dropdown_choices(chats), value=current_chat_id
    )


def create_new_chat():
    chats = load_chats()
    new_id = str(uuid.uuid4())
    chats[new_id] = {"title": "New Chat", "messages": []}
    save_chats(chats)
    return new_id, [], gr.update(choices=get_dropdown_choices(chats), value=new_id), "New Chat"


def switch_chat(selected_id):
    chats = load_chats()
    if selected_id and selected_id in chats:
        return selected_id, chats[selected_id]["messages"], chats[selected_id]["title"]
    return selected_id, [], ""


def rename_chat(selected_id, new_title):
    chats = load_chats()
    if selected_id in chats and new_title and new_title.strip():
        chats[selected_id]["title"] = new_title.strip()
        save_chats(chats)
    return gr.update(choices=get_dropdown_choices(chats), value=selected_id)


def delete_chat(selected_id):
    chats = load_chats()
    if selected_id in chats:
        del chats[selected_id]
    if not chats:
        new_id = str(uuid.uuid4())
        chats[new_id] = {"title": "New Chat", "messages": []}
    save_chats(chats)

    first_id = list(chats.keys())[0]
    return (
        first_id,
        chats[first_id]["messages"],
        gr.update(choices=get_dropdown_choices(chats), value=first_id),
        chats[first_id]["title"],
    )


def transcribe_audio(audio_file_path, model_name):
    if not audio_file_path:
        return "⚠️ Please record or upload an audio file first."
    try:
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model=model_name,
                response_format="text",
            )
        return transcription
    except Exception as e:
        return f"⚠️ **STT Error:** {str(e)}"


def generate_speech(text, model_name, voice_name):
    if not text or not text.strip():
        return None, "⚠️ Please enter some text first."
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        response = client.audio.speech.create(
            model=model_name,
            voice=voice_name.lower(),
            input=text,
            response_format="wav",
        )
        response.write_to_file(tmp.name)
        return tmp.name, "✅ Speech generated successfully!"
    except Exception as e:
        return None, f"⚠️ TTS Error: {str(e)}"


def run_voice_agent(audio_file_path, agent_goal, agent_history):
    if not audio_file_path:
        return "", "", None, agent_history

    try:
        with open(audio_file_path, "rb") as file:
            user_speech_text = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file.read()),
                model="whisper-large-v3-turbo",
                language="en",
                response_format="text",
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

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
        )
        agent_reply_text = completion.choices[0].message.content

        updated_history = list(agent_history) + [
            {"role": "user", "content": user_speech_text.strip()},
            {"role": "assistant", "content": agent_reply_text},
        ]

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        response = client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="troy",
            input=agent_reply_text,
            response_format="wav",
        )
        response.write_to_file(tmp.name)

        return user_speech_text.strip(), agent_reply_text, tmp.name, updated_history

    except Exception as e:
        return "Error", f"⚠️ Agent Error: {str(e)}", None, agent_history


def reset_agent_turn():
    return None


VAD_AND_TURN_JS = """
() => {
    let audioCtx = null;
    let analyser = null;
    let micStream = null;
    let vadInterval = null;
    let isAgentSpeaking = false;

    window.startHandsFreeLoop = async () => {
        const container = document.getElementById('agent-mic-box');
        if (!container) return;

        const clearBtn = container.querySelector('button[aria-label="Clear"]');
        if (clearBtn) clearBtn.click();

        setTimeout(async () => {
            const recordBtn = container.querySelector('button.record-button, button[aria-label*="Record"]');
            if (recordBtn && !isAgentSpeaking) {
                recordBtn.click();
            }

            try {
                if (!micStream) {
                    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const source = audioCtx.createMediaStreamSource(micStream);
                    analyser = audioCtx.createAnalyser();
                    analyser.fftSize = 512;
                    source.connect(analyser);
                }

                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                let hasSpoken = false;
                let silenceStart = null;

                if (vadInterval) clearInterval(vadInterval);
                vadInterval = setInterval(() => {
                    if (isAgentSpeaking) return;

                    analyser.getByteFrequencyData(dataArray);
                    let sum = 0;
                    for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
                    let avgVolume = sum / dataArray.length;

                    if (avgVolume > 18) {
                        hasSpoken = true;
                        silenceStart = null;
                    } else if (hasSpoken) {
                        if (!silenceStart) {
                            silenceStart = Date.now();
                        } else if (Date.now() - silenceStart > 1500) {
                            clearInterval(vadInterval);
                            const stopBtn = container.querySelector('button.stop-button, button[aria-label*="Stop"]');
                            if (stopBtn) {
                                isAgentSpeaking = true;
                                stopBtn.click();
                            }
                        }
                    }
                }, 100);
            } catch (err) {
                console.error("Mic VAD error:", err);
            }
        }, 300);
    };

    window.onAgentFinishedSpeaking = () => {
        isAgentSpeaking = false;
        const autoToggle = document.querySelector('#hands-free-toggle input[type="checkbox"]');
        if (autoToggle && autoToggle.checked) {
            window.startHandsFreeLoop();
        }
    };
}
"""

initial_chats = load_chats()
initial_id = list(initial_chats.keys())[0]

with gr.Blocks(title="⚡ Groq Multimodal Suite", theme=gr.themes.Citrus(), js=VAD_AND_TURN_JS) as demo:
    with gr.Tabs():
        with gr.TabItem("💬 Chat & Manager"):
            current_chat_state = gr.State(value=initial_id)
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("## 💬 Chat Manager")
                    new_chat_btn = gr.Button("+ New Chat", variant="primary")

                    chat_selector = gr.Dropdown(
                        choices=get_dropdown_choices(initial_chats),
                        value=initial_id,
                        label="Saved Conversations",
                        interactive=True,
                    )

                    with gr.Group():
                        title_input = gr.Textbox(
                            value=initial_chats[initial_id]["title"],
                            label="Edit Chat Title",
                            placeholder="Enter new title...",
                        )
                        with gr.Row():
                            rename_btn = gr.Button("Save Title", size="sm")
                            delete_btn = gr.Button("Delete Chat", variant="stop", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("### ⚙️ Model & Search")
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "openai/gpt-oss-120b",
                            "openai/gpt-oss-20b",
                        ],
                        value="openai/gpt-oss-120b",
                        label="Groq Model",
                    )
                    search_checkbox = gr.Checkbox(label="Enable DuckDuckGo Search", value=False)
                    system_prompt_box = gr.Textbox(
                        value="You are a helpful, laid-back AI assistant.",
                        label="Custom System Prompt",
                        lines=3,
                    )

                with gr.Column(scale=3):
                    gr.Markdown("# ⚡ Groq Local Assistant")
                    chatbot = gr.Chatbot(
                        value=initial_chats[initial_id]["messages"],
                        height=560,
                        label="Conversation",
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your message and press Enter...",
                            show_label=False,
                            scale=5,
                            container=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.TabItem("🤖 Voice Agent"):
            agent_history_state = gr.State(value=[])
            gr.Markdown("# 🤖 Hands-Free Voice Agent (Auto-Pause & Turn-Taking)")
            gr.Markdown(
                "Set your agent's goal, toggle **Hands-Free Auto-Listen**, and start talking. "
                "It automatically stops after a 1.5s pause, finishes its entire spoken reply without interruption, "
                "and then listens for your next turn."
            )
            with gr.Row():
                with gr.Column():
                    agent_goal_input = gr.Textbox(
                        label="Agent Goal / System Prompt",
                        value="Inform the user that they got an A+ in all their classes with an enthusiastic, proud, and celebratory tone. Keep responses conversational and concise.",
                        lines=3,
                    )
                    hands_free_checkbox = gr.Checkbox(
                        label="🎙️ Hands-Free Auto-Listen (Auto-stop on pause & resume after AI finishes speaking)",
                        value=True,
                        elem_id="hands-free-toggle",
                    )
                    start_convo_btn = gr.Button("▶️ Start Hands-Free Conversation", variant="primary")
                    clear_convo_btn = gr.Button("🔄 Reset Conversation Memory", size="sm")

                    agent_audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Your Microphone (Auto-stops after pause)",
                        elem_id="agent-mic-box",
                    )
                with gr.Column():
                    agent_stt_output = gr.Textbox(label="1. What You Said (Whisper Turbo)")
                    agent_text_output = gr.Textbox(label="2. Agent Reply (GPT OSS 20B)", lines=4)
                    agent_audio_output = gr.Audio(
                        label="3. Spoken Response (Orpheus - Troy)",
                        type="filepath",
                        autoplay=True,
                        elem_id="agent-audio-out",
                    )

        with gr.TabItem("🎙️ Speech-to-Text (STT)"):
            gr.Markdown("# 🎙️ Groq Whisper Speech-to-Text")
            gr.Markdown("Record your voice or upload an audio file to quickly transcribe it into text.")
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(type="filepath", label="Audio Input (Record or Upload)")
                    stt_model_dropdown = gr.Dropdown(
                        choices=["whisper-large-v3-turbo", "whisper-large-v3"],
                        value="whisper-large-v3-turbo",
                        label="Whisper Model",
                    )
                    stt_btn = gr.Button("Transcribe Audio", variant="primary")
                with gr.Column():
                    stt_output = gr.Textbox(label="Transcription Result", lines=12)

        with gr.TabItem("🔊 Text-to-Speech (TTS)"):
            gr.Markdown("# 🔊 Groq Orpheus Text-to-Speech")
            gr.Markdown(
                "Convert text into natural speech. *(Tip: Use bracketed tags like `[cheerful]` or `[whisper]` for voice styling)*"
            )
            with gr.Row():
                with gr.Column():
                    tts_input = gr.Textbox(
                        label="Text to Convert",
                        placeholder="Type something here... e.g. [cheerful] Hello! Welcome to the app.",
                        lines=4,
                    )
                    tts_model_dropdown = gr.Dropdown(
                        choices=["canopylabs/orpheus-v1-english"],
                        value="canopylabs/orpheus-v1-english",
                        label="TTS Model",
                    )
                    tts_voice_dropdown = gr.Dropdown(
                        choices=["troy", "autumn", "diana", "hannah", "austin", "daniel"],
                        value="troy",
                        label="Voice Persona",
                    )
                    tts_btn = gr.Button("Generate Speech", variant="primary")
                with gr.Column():
                    tts_output = gr.Audio(label="Generated Audio Output", type="filepath")
                    tts_status = gr.Textbox(label="Status / Error Log", interactive=False)

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, current_chat_state, system_prompt_box, model_dropdown, search_checkbox],
        outputs=[msg_input, chatbot, chat_selector],
    )
    send_btn.click(
        fn=respond,
        inputs=[msg_input, current_chat_state, system_prompt_box, model_dropdown, search_checkbox],
        outputs=[msg_input, chatbot, chat_selector],
    )
    new_chat_btn.click(
        fn=create_new_chat,
        inputs=[],
        outputs=[current_chat_state, chatbot, chat_selector, title_input],
    )
    chat_selector.change(
        fn=switch_chat,
        inputs=[chat_selector],
        outputs=[current_chat_state, chatbot, title_input],
    )
    rename_btn.click(
        fn=rename_chat,
        inputs=[current_chat_state, title_input],
        outputs=[chat_selector],
    )
    delete_btn.click(
        fn=delete_chat,
        inputs=[current_chat_state],
        outputs=[current_chat_state, chatbot, chat_selector, title_input],
    )

    start_convo_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { window.startHandsFreeLoop(); }",
    )

    clear_convo_btn.click(
        fn=lambda: ([], "", "", None),
        inputs=[],
        outputs=[agent_history_state, agent_stt_output, agent_text_output, agent_audio_output],
    )

    agent_audio_input.stop_recording(
        fn=run_voice_agent,
        inputs=[agent_audio_input, agent_goal_input, agent_history_state],
        outputs=[agent_stt_output, agent_text_output, agent_audio_output, agent_history_state],
    )

    agent_audio_output.stop(
        fn=reset_agent_turn,
        inputs=[],
        outputs=[agent_audio_input],
        js="() => { window.onAgentFinishedSpeaking(); }",
    )

    stt_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, stt_model_dropdown],
        outputs=[stt_output],
    )
    tts_btn.click(
        fn=generate_speech,
        inputs=[tts_input, tts_model_dropdown, tts_voice_dropdown],
        outputs=[tts_output, tts_status],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )

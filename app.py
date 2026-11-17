import os
import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKERS = ["Ryan", "Serena", "Vivian", "Aiden", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee"]
LANGUAGES = ["Russian", "English", "Japanese", "Korean", "Chinese", "Auto"]

print(f"[INFO] Запуск модели {MODEL_ID}...")
tts_model = Qwen3TTSModel.from_pretrained(
    MODEL_ID,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    attn_implementation="sdpa",
)
print(f"[INFO] Модель успешно загрузилась")

def synthesize_speech(text: str, language: str, speaker: str, instruct: str) -> str:
    if not text or not text.strip():
        raise gr.Error("Поле текста не может быть пустым!")

    wavs, sr = tts_model.generate_custom_voice(
        text=text.strip(),
        language=language,
        speaker=speaker,
        instruct=instruct.strip() if instruct and instruct.strip() else None,
    )

    output_path = "temp_output.wav"
    sf.write(output_path, wavs[0], sr)
    return output_path


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS Studio") as demo:
        gr.Markdown("Qwen3-TTS Audio Production Studio")

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Текст для озвучки (Text)",
                    lines=4,
                    placeholder="Введите текст, который нужно озвучить...",
                )
                with gr.Row():
                    lang_input = gr.Dropdown(
                        choices=LANGUAGES, value="Russian", label="Язык (Language)"
                    )
                    speaker_input = gr.Dropdown(
                        choices=SPEAKERS, value="Ono_Anna", label="Голос (Speaker)"
                    )

                instruct_input = gr.Textbox(
                    label="Инструкция интонации (Instruction)",
                    lines=2,
                    placeholder="Например: Speak slowly, in a soft, calm tone...",
                )
                generate_btn = gr.Button("🎙 Сгенерировать", variant="primary")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Результат", type="filepath")

        generate_btn.click(
            fn=synthesize_speech,
            inputs=[text_input, lang_input, speaker_input, instruct_input],
            outputs=audio_output,
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=8000)
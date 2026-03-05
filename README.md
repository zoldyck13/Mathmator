# Mathmator: The AI-Powered Local Manim Copilot 🤖

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Local LLM](https://img.shields.io/badge/LLM-Llama--3--8B-orange)
![Manim](https://img.shields.io/badge/Rendered_with-Manim-ececec?logo=python)

**Mathmator** is a cutting-edge, 100% offline, local AI assistant designed to generate, edit, and render high-quality mathematical and scientific animations using the **Manim** engine.

Whether you want to visualize a neural network, plot complex calculus functions, or animate physics concepts, just tell Mathmator what you want, and it will write the code and render the video for you.

---

## 🌟 Key Features

* **🧠 100% Local AI:** Powered by a customized `Llama-3-8B` model with a highly specialized LoRA fine-tuned on golden Manim examples. No API keys, no internet required (after initial setup), absolute privacy.
* **🎬 Cinematic Auto-Design:** Built-in "Mega-Prompt" architecture ensures generated animations follow professional design guidelines (titles, colors, sequential reveals, pacing).
* **🔄 Context-Aware Copilot (`edit`):** Don't like the color? Want to make it slower? Just type your edit. Mathmator remembers the previous code, modifies it, and re-renders intelligently.
* **🎙️ Voice Mode (`voice`):** Talk to Mathmator! Uses Next-Gen **OpenAI Whisper** for highly accurate local Speech-to-Text (STT) and **Silero Neural TTS** for a natural, human-like voice response. 
* **🛡️ Syntax Shield & Auto-Patcher:** Automatically patches common LLM hallucinations (like unclosed parentheses, deprecated methods, or infinite loops) before sending the code to Manim.
* **⚡ GPU Acceleration:** Seamlessly detects and utilizes your NVIDIA GPU via `llama.cpp` (CUDA) for lightning-fast code generation.

---

## 🛠️ Prerequisites

Before installing Mathmator, you need to have the Manim Community dependencies installed on your system:
* **FFmpeg** (for video rendering)
* **LaTeX** (for rendering math formulas)
* **NVIDIA GPU (Optional but highly recommended)** for fast LLM inference and TTS/STT.

*(For full Manim installation instructions, visit the [Manim Docs](https://docs.manim.community/en/stable/installation.html))*

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/zoldyck13/Mathmator.git
cd Mathmator
```

**2. Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**4. **Install** `llama-cpp-python` with **CUDA (For GPU Users)**:**
If you have an NVIDIA GPU, run this to enable hardware acceleration:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

**5. Download the Models:**
Place your base Llama-3 GGUF model and your custom LoRA adapter in the root directory:

* `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`

* `mathmator_lora.gguf`

_____
## Usage

Mathmator uses a sleek Typer-based CLI.

**1. Generate an Animation from Scratch**

Use the `animate` command to describe your concept. Use `-q` to set quality (`low`, `medium`, `high`).

**2. The Interactive Copilot Loop**

After generating an animation, Mathmator enters an interactive loop. It will ask:
`✍️ Enter your edit for this video (or type 'exit' to quit):`
You can simply type:

<p align="center">"Change the sun's color to bright yellow and make the planets smaller."</p>


**3. Edit the Last Animation Directly**
If you closed the app and want to modify the last generated video:
```bash
python mathmator.py edit "Make the animation slower and add a title saying 'Orbital Mechanics'" -q medium
```

**4. Voice Mode**
Run the voice assistant mode to talk directly to Mathmator.
```bash
python mathmator.py voice -q low
```

* Press ENTER to activate the mic.

* Say: "Mathmator animate a sine wave and a cosine wave."

* Listen to the AI response as it renders your video!

____
## Under the Hood

1. **Voice Input**: `SpeechRecognition` + `OpenAI Whisper (base.en)` transcribes your prompt locally.

2. **Prompt Engineering**: Your prompt is injected into a specialized design guideline template.

3. **LLM Generation**: `llama.cpp` runs the Llama-3 model + LoRA to generate Python Manim code.

4. **Auto-Patching**: Regex scripts clean up the generated code, fix deprecated `.set_height()`, replace 3D `x_range` with `u_range`, and close missing brackets.

5. **Rendering**: The script invokes the Manim CLI subprocess to render the video.

6. **Voice Output**: `Silero TTS` informs you of the progress with a high-fidelity neural voice.

___
## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

___

📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

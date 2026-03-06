import os
# CRITICAL: Prevent OpenMP thread conflicts between PyTorch and llama.cpp
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import typer
import subprocess
import shutil
import re
import speech_recognition as sr
import sys
import torch
import sounddevice as sd
import zipfile
import warnings
from enum import Enum
from llama_cpp import Llama

# Suppress specific warnings from Whisper/Torch to keep output clean
warnings.filterwarnings("ignore", message="Performing inference on CPU when CUDA is available")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Prevent PyTorch from hogging CPU threads
torch.set_num_threads(4)

app = typer.Typer()

# --- MODEL PATHS & URLS (BASE + LORA ONLY) ---
BASE_MODEL_PATH = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
BASE_MODEL_URL = "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"

LORA_PATH = "./mathmator_lora.gguf"
LORA_ZIP_PATH = "./mathmator_lora.gguf.zip"
LORA_ZIP_URL = "https://huggingface.co/Alisaadmotar/Mathmator-Llama3-LoRA/resolve/main/mathmator_lora.gguf.zip"

LATEST_CODE_FILE = "latest_mathmator_code.py"
LAST_ERROR_LOG = ""  # Memory for the last crash

# --- 🛑 CRITICAL MANIM CE EXPLICIT RULES 🛑 ---
MANIM_RULES = (
    "\n--- CRITICAL MANIM CE 2024 SYNTAX ---\n"
    "You MUST use Manim Community Edition syntax. Old ManimGL syntax will crash!\n"
    "1. Background: `self.camera.background_color = BLACK`\n"
    "2. Axes: `axes = Axes(x_range=[-10, 10, 1], y_range=[-5, 5, 1], x_length=10, y_length=6)`\n"
    "3. Plotting: `graph = axes.plot(lambda x: np.sin(x), color=BLUE)`\n"
    "4. Labels: `labels = axes.get_axis_labels(x_label='x', y_label='y')`\n"
    "5. Title: `title = Title('Your Title Here')`\n"
    "6. Animations: Use `Create()` instead of `ShowCreation()`.\n"
    "7. Coordinates: ALL coordinates MUST be 3D arrays like `[x, y, 0]`. NEVER use 2D `[x, y]`! Example: `move_to([1, 2, 0])`\n"
    "8. Keep loops short (max 3 iterations).\n"
    "9. TEXT WRAPPING (CRITICAL): NEVER write long sentences in Text(). For long descriptions, ALWAYS use Paragraph('line1', 'line2', line_spacing=0.5) or use Text('...', font_size=20). If a sentence is more than 8 words, break it into multiple lines.\n"
)

class SuppressStderr:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)
        os.close(self.save_fd)

def check_gpu():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

# --- DYNAMIC HARDWARE DETECTION & DOWNLOAD LOGIC ---
# Download models if they don't exist
if not os.path.exists(BASE_MODEL_PATH):
    typer.secho("Base model not found. Downloading Meta-Llama-3-8B-Instruct-Q4_K_M.gguf...", fg=typer.colors.YELLOW)
    torch.hub.download_url_to_file(BASE_MODEL_URL, BASE_MODEL_PATH)

if not os.path.exists(LORA_PATH):
    if not os.path.exists(LORA_ZIP_PATH):
        typer.secho("LoRA adapter not found. Downloading mathmator_lora.gguf.zip...", fg=typer.colors.YELLOW)
        try:
            torch.hub.download_url_to_file(LORA_ZIP_URL, LORA_ZIP_PATH)
        except Exception as e:
            typer.secho(f"Download failed: {e}", fg=typer.colors.RED)
            sys.exit(1)
            
    typer.secho("Extracting mathmator_lora.gguf...", fg=typer.colors.CYAN)
    try:
        with zipfile.ZipFile(LORA_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        if os.path.exists(LORA_ZIP_PATH):
            os.remove(LORA_ZIP_PATH)
    except Exception as e:
        typer.secho(f"Extraction failed: {e}", fg=typer.colors.RED)
        sys.exit(1)

# Configure hardware settings dynamically
has_gpu = check_gpu()

if has_gpu:
    typer.secho("🤖 Mathmator: GPU Detected! 🚀 (Using Base + LoRA with mmap=False to prevent Segfault)", fg=typer.colors.GREEN, bold=True)
    gpu_layers = -1
    use_mmap = False      # CRITICAL: Prevents GPU Segfault with LoRA
    max_tokens_val = 1500 # GPU has memory for more tokens
    ctx_size = 2048
else:
    typer.secho("🤖 Mathmator: CPU Detected! 🐢 (Using Base + LoRA with mmap=True to save RAM)", fg=typer.colors.YELLOW, bold=True)
    gpu_layers = 0
    use_mmap = True       # CRITICAL: Prevents OOM/Killed on CPU
    max_tokens_val = 700  # CPU gets smaller generation to save RAM
    ctx_size = 1024

llm_kwargs = {
    "model_path": BASE_MODEL_PATH,
    "lora_path": LORA_PATH,
    "n_ctx": ctx_size,
    "n_gpu_layers": gpu_layers,
    "n_threads": 4,
    "n_batch": 256 if has_gpu else 128,
    "use_mmap": use_mmap,
    "verbose": False
}

# --- INITIALIZE THE ADAPTIVE MODEL ---
with SuppressStderr():
    llm = Llama(**llm_kwargs)


# --- AUDIO MODELS ON CPU ---
device_audio = torch.device('cpu')
model_file = 'silero_v3_en.pt'

if not os.path.isfile(model_file):
    typer.secho("Downloading High-Quality Offline Voice Model (One time only)...", fg=typer.colors.YELLOW)
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt', model_file)

try:
    with SuppressStderr():
        tts_model = torch.package.PackageImporter(model_file).load_pickle("tts_models", "model")
        tts_model.to(device_audio)
except Exception as e:
    typer.secho(f"Failed to load TTS model: {e}", fg=typer.colors.RED)

class Quality(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

def speak(text: str, use_voice: bool = True):
    typer.secho(f"🤖 Mathmator: {text}", fg=typer.colors.CYAN)
    if use_voice:
        try:
            clean_text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)
            if not clean_text.endswith(('.', '!', '?')):
                clean_text += '.'
                
            audio_tensor = tts_model.apply_tts(
                text=clean_text,
                speaker='en_0',
                sample_rate=48000
            )
            sd.play(audio_tensor.cpu().numpy(), 48000)
            sd.wait()
        except Exception as e:
            typer.secho(f"TTS Audio Error: {e}", fg=typer.colors.RED)

def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        typer.secho("\n🎙️ Listening... (Say your command now)", fg=typer.colors.GREEN)
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            typer.secho("⏳ Processing speech with Offline Whisper on CPU...", fg=typer.colors.MAGENTA)
            
            command = recognizer.recognize_whisper(audio, model="base.en", load_options={"device": "cpu"}, fp16=False).strip()
            command = re.sub(r'[^\w\s]', '', command).lower()
            
            typer.secho(f"🗣️ You said: {command}", fg=typer.colors.YELLOW)
            return command
            
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except Exception as e:
            typer.secho(f"Speech recognition error: {e}", fg=typer.colors.RED)
            return ""

def generate_dynamic_speech(command: str) -> str:
    typer.secho("🧠 Formulating dynamic response...", fg=typer.colors.MAGENTA)
    prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\nSummarize what you will do in one short conversational sentence without any code. The user asked: {command}\n\n"
        "### Response:\nSure, I will"
    )
    try:
        output = llm(
            prompt,
            max_tokens=25,
            stop=["\n", ".", "```", "<|end_of_text|>", "<|eot_id|>"],
            temperature=0.5,
            echo=False
        )
        text = output["choices"][0]["text"].strip()
        if text:
            return "Sure, I will " + text + "."
        else:
            return "Right away, working on that for you."
    except Exception:
        return "Right away, working on that for you."

def explain_error_speech(error_log: str) -> str:
    typer.secho("🧠 Analyzing the crash error...", fg=typer.colors.MAGENTA)
    clean_log = re.sub(r'\x1b\[[0-9;]*m', '', error_log)
    short_error = clean_log[-600:].strip()
    prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\nThe python code crashed with this error:\n{short_error}\n"
        "Explain what went wrong simply and briefly to the user in one sentence without writing code.\n\n"
        "### Response:\nIt seems that"
    )
    try:
        output = llm(
            prompt,
            max_tokens=30,
            stop=["\n", ".", "```", "<|end_of_text|>", "<|eot_id|>"],
            temperature=0.4,
            echo=False
        )
        text = output["choices"][0]["text"].strip()
        if text:
            return "It seems that " + text + ". Would you like me to try and fix it?"
        else:
            return "It seems there is a syntax error in the code. Would you like me to try and fix it?"
    except Exception:
        return "It seems there is a syntax error in the code. Would you like me to try and fix it?"

def process_and_render(prompt_text: str, safe_topic_name: str, quality: Quality, keep_code: bool, prefill: str, use_voice: bool = False):
    global LAST_ERROR_LOG
    temp_file = f"{safe_topic_name}_scene.py"
    
    try:
        typer.secho("\n🧠 Mathmator writing now ...\n", fg=typer.colors.BLUE)
        print(prefill, end="", flush=True)

        output = llm(
            prompt_text, 
            max_tokens=max_tokens_val, 
            stop=["<|end_of_text|>", "<|eot_id|>", "###", "```", "class Concept", "\nclass ", "\ndef ", "if __name__", "\n    def ", "\nConceptScene", "\n# Run"], 
            temperature=0.1,   
            repeat_penalty=1.05,  
            stream=True
        )
        
        manim_code = prefill
        for chunk in output:
            text = chunk["choices"][0].get("text", "")
            print(text, end="", flush=True) 
            manim_code += text
        print("\n")

        # 🧹 SMART CLEANUP & HEALING SHIELDS
        manim_code = manim_code.replace("```python", "").replace("```", "").strip()
        manim_code = re.sub(r'ApplyMethod\((.*?)\)', r'\1', manim_code)
        manim_code = manim_code.replace("ShowCreation", "Create")
        
        # Kill runaway execution code
        manim_code = re.sub(r'\n[ \t]*[a-zA-Z0-9_]+\s*=\s*ConceptScene\(\).*', '', manim_code, flags=re.DOTALL)
        manim_code = re.sub(r'\n[ \t]*scene\.render\(\).*', '', manim_code, flags=re.DOTALL)
        manim_code = re.sub(r'\n[ \t]*if __name__\s*==.*', '', manim_code, flags=re.DOTALL)
        manim_code = re.sub(r'\n[ \t]*ConceptScene\(\).*', '', manim_code, flags=re.DOTALL)
        manim_code = re.sub(r'\n[ \t]*# Run the.*', '', manim_code, flags=re.DOTALL)
        manim_code = re.sub(r'Cone\(\s*radius=', 'Cone(base_radius=', manim_code)

        # 🛡️ THE ANIMATION HEALER: Fixes complex Updaters by enforcing .animate syntax
        manim_code = re.sub(r'UpdateFromFunc\s*\(\s*lambda\s+[a-zA-Z0-9_]+\s*:\s*[a-zA-Z0-9_]+\.(.*?)\s*,\s*([a-zA-Z0-9_]+).*?\)', r'\2.animate.\1', manim_code)

        # 🛡️ SMART REGEX SHIELDS: Heal the code instead of deleting the line to prevent NameErrors!
        manim_code = re.sub(r'self\.set_background_color\((.*?)\)', r'self.camera.background_color = \1', manim_code)
        
        # 🛡️ THE VGROUP HEALER V3.0: Surgical precision. Fixes ONLY List Comprehensions inside VGroup!
        # Matches: VGroup(*[Create(mob) for mob in inputs]) -> VGroup(*[mob for mob in inputs])
        manim_code = re.sub(r'(VGroup\s*\(\s*\*?\s*\[\s*)(?:Create|Write|FadeIn|FadeOut)\s*\(\s*([^)]+)\s*\)(\s*for\s*[^\]]+\]\s*\))', r'\1\2\3', manim_code)

        # If it assigns to a variable, replace the bad call with an empty VGroup so Python doesn't crash on NameError later
        manim_code = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*.*\.set_axis_labels\(.*\)', r'\1 = VGroup() # Auto-healed bad label', manim_code)
        manim_code = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*.*\.set_title\(.*\)', r'\1 = VGroup() # Auto-healed bad title', manim_code)
        manim_code = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*Title\(.*\)', r'\1 = VGroup() # Auto-healed bad Title obj', manim_code)
        
        # If it's a standalone call, safely comment it out
        manim_code = re.sub(r'(^[ \t]*.*\.set_axis_labels\(.*\))', r'# \1 (Auto-removed)', manim_code, flags=re.MULTILINE)
        manim_code = re.sub(r'(^[ \t]*.*\.set_title\(.*\))', r'# \1 (Auto-removed)', manim_code, flags=re.MULTILINE)

        if "Surface(" in manim_code:
            parts = manim_code.split("Surface(")
            for i in range(1, len(parts)):
                parts[i] = parts[i].replace("x_range", "u_range").replace("y_range", "v_range")
            manim_code = "Surface(".join(parts)

        manim_code = manim_code.rstrip()
        if not re.search(r'self\.wait\([^)]*\)\s*$', manim_code):
            manim_code += "\n        self.wait(2)"

        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(manim_code)
            
        with open(LATEST_CODE_FILE, "w", encoding="utf-8") as f:
            f.write(manim_code)

        if quality == Quality.medium:
            quality_flag = "-qm"
            quality_folder = "720p30"
        elif quality == Quality.high:
            quality_flag = "-qh"
            quality_folder = "1080p60"
        else:
            quality_flag = "-ql"
            quality_folder = "480p15"

        speak(f"Rendering at {quality.value} quality. This might take a minute, please wait...", use_voice)

        # 🚀 THE MAGIC: REAL-TIME PROGRESS BAR PIPING 🚀
        env = os.environ.copy()
        env["FORCE_COLOR"] = "1"  # Forces Manim's rich progress bar to render even in background piping

        process = subprocess.Popen(
            ["manim", quality_flag, temp_file, "ConceptScene", "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0, # Unbuffered, so we catch the progress bar updates (\r) instantly
            env=env
        )
        
        output_bytes = bytearray()
        while True:
            char = process.stdout.read(1)
            if not char and process.poll() is not None:
                break
            if char:
                sys.stdout.buffer.write(char) # Print strictly to terminal
                sys.stdout.buffer.flush()
                output_bytes.extend(char)     # Save safely in the background
                
        if process.returncode != 0:
            error_text = output_bytes.decode('utf-8', errors='replace')
            raise subprocess.CalledProcessError(process.returncode, process.args, output=error_text, stderr=error_text)

        video_path = os.path.join("media", "videos", f"{safe_topic_name}_scene", quality_folder, "ConceptScene.mp4")
        final_video_name = f"{safe_topic_name}.mp4"

        if os.path.exists(video_path):
            shutil.copy(video_path, final_video_name)
            LAST_ERROR_LOG = ""  # Clear error on success
            speak("Finished successfully! The video is ready.", use_voice)
        else:
            speak("Video was generated but couldn't be found.", use_voice)

    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else e.stdout
        LAST_ERROR_LOG = error_output  # Save the error for the AI to fix later
        typer.secho(error_output, fg=typer.colors.RED)
        explanation = explain_error_speech(error_output)
        speak(explanation, use_voice)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
    finally:
        if not keep_code and os.path.exists(temp_file):
            os.remove(temp_file)

def interactive_edit_loop(quality: Quality, keep_code: bool, use_voice: bool = False):
    global LAST_ERROR_LOG
    while True:
        typer.secho("\n" + "="*60, fg=typer.colors.MAGENTA)
        instruction = typer.prompt("✍️  Enter your edit for this video (or type 'exit' to quit)")
        
        if instruction.strip().lower() in ["exit", "quit", "stop"]:
            speak("Goodbye! Exiting Mathmator.", use_voice)
            raise typer.Exit()
            
        if not os.path.exists(LATEST_CODE_FILE):
            typer.secho("Error: No previous code found to edit.", fg=typer.colors.RED)
            continue
            
        with open(LATEST_CODE_FILE, "r", encoding="utf-8") as f:
            current_code = f.read()
            
        safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', instruction)
        safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
        if not safe_topic_name:
            safe_topic_name = "interactive_edit"
        
        match = re.search(r'class ConceptScene\((.*?)\):', current_code)
        scene_type = match.group(1) if match else "Scene"
        
        prefill_code = f"from manim import *\nimport numpy as np\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
        
        error_context = f"\n\nCRITICAL - The previous code crashed with this error:\n{LAST_ERROR_LOG[-400:]}\nPlease fix the error based on this." if LAST_ERROR_LOG else ""
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nMake this code more professional based on the user's edit: {instruction}{error_context}\nEnsure elegant animations and CRITICALLY ensure all Python parentheses are closed properly.\n{MANIM_RULES}\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
        
        process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice)

@app.command()
def animate(topic: str = typer.Argument(...),
            quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
            keep_code: bool = typer.Option(False, "--keep-code", "-k")):
    
    safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', topic)
    safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
    
    scene_type = "ThreeDScene" if "3d" in topic.lower() else "Scene"
        
    prefill_code = f"from manim import *\nimport numpy as np\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
    
    enhanced_instruction = (
        f"Design and code a highly professional, cinematic Manim educational animation about: '{topic}'.\n"
        "You are an expert director and mathematician (like 3Blue1Brown). Think deeply about the concept and intelligently expand on the user's idea by adding necessary mathematical context, beautiful visual elements, and logical step-by-step explanations.\n"
        "Follow these strict design guidelines:\n"
        "1. Professional Layout: Arrange elements elegantly. Use appropriate scaling and positioning so nothing overlaps the screen edges.\n"
        "2. Creative Typography: Be creative and organic with text. Do not restrict yourself to always adding a static title at the top. Place labels and explanations dynamically where they make the most sense visually.\n"
        "3. Color Palette: Use professional Manim colors (BLUE_E, TEAL, YELLOW, RED) to distinguish different components.\n"
        "4. Sequential Animation: Do not show everything at once. Animate step-by-step (e.g., draw axes, plot graph, add labels).\n"
        "5. Flow: Wait for 1 or 2 seconds between major animations using self.wait().\n"
        f"{MANIM_RULES}\n"
        "Write only the python code."
    )
    
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{enhanced_instruction}\n\n### Response:\n```python\n{prefill_code}"
    
    process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=False)
    interactive_edit_loop(quality, keep_code, use_voice=False)

@app.command()
def edit(instruction: str = typer.Argument(...),
         quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
         keep_code: bool = typer.Option(False, "--keep-code", "-k")):
    
    global LAST_ERROR_LOG
    if not os.path.exists(LATEST_CODE_FILE):
        typer.secho("No previous animation found to edit! Run 'animate' first.", fg=typer.colors.RED)
        raise typer.Exit()
        
    with open(LATEST_CODE_FILE, "r", encoding="utf-8") as f:
        current_code = f.read()
        
    safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', instruction)
    safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
    
    match = re.search(r'class ConceptScene\((.*?)\):', current_code)
    scene_type = match.group(1) if match else "Scene"
    
    prefill_code = f"from manim import *\nimport numpy as np\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
    error_context = f"\n\nCRITICAL - The previous code crashed with this error:\n{LAST_ERROR_LOG[-400:]}\nPlease fix the error based on this." if LAST_ERROR_LOG else ""
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nMake this code more professional based on the user's edit: {instruction}{error_context}\nEnsure elegant animations and CRITICALLY ensure all Python parentheses are closed properly.\n{MANIM_RULES}\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
    
    process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=False)
    interactive_edit_loop(quality, keep_code, use_voice=False)

@app.command()
def voice(quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
          keep_code: bool = typer.Option(False, "--keep-code", "-k")):
    
    global LAST_ERROR_LOG
    speak("Voice mode activated. Press Enter when you are ready to speak.", use_voice=True)
    
    while True:
        typer.secho("\n" + "="*60, fg=typer.colors.MAGENTA)
        try:
            input("🔴 Press ENTER to start speaking... (or Ctrl+C to quit) ")
        except KeyboardInterrupt:
            print()
            speak("Goodbye! Shutting down voice mode.", use_voice=True)
            break
            
        command = listen_command()
        
        if not command:
            continue
            
        if any(word in command for word in ["exit", "stop", "quit", "goodbye"]):
            speak("Goodbye! Shutting down voice mode.", use_voice=True)
            break
            
        edit_keywords = ["change", "edit", "update", "modify", "make", "add", "remove", "delete", "fix", "replace", "color", "faster", "slower", "yes", "sure"]
        first_words = command.split()[:3]
        
        is_edit = False
        if os.path.exists(LATEST_CODE_FILE):
            if any(kw in first_words for kw in edit_keywords) or LAST_ERROR_LOG:
                is_edit = True
                
        if command.startswith("animate "):
            is_edit = False
            command = command.replace("animate ", "", 1).strip()
        elif command.startswith("edit "):
            is_edit = True
            command = command.replace("edit ", "", 1).strip()
            
        dynamic_response = generate_dynamic_speech(command)
        speak(dynamic_response, use_voice=True)
            
        if is_edit:
            with open(LATEST_CODE_FILE, "r", encoding="utf-8") as f:
                current_code = f.read()
                
            safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', command)
            safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
            if not safe_topic_name:
                safe_topic_name = "interactive_edit"
            
            match = re.search(r'class ConceptScene\((.*?)\):', current_code)
            scene_type = match.group(1) if match else "Scene"
            
            prefill_code = f"from manim import *\nimport numpy as np\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
            error_context = f"\n\nCRITICAL - The previous code crashed with this error:\n{LAST_ERROR_LOG[-400:]}\nPlease fix the error based on this." if LAST_ERROR_LOG else ""
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nMake this code more professional based on the user's edit: {command}{error_context}\nEnsure elegant animations and CRITICALLY ensure all Python parentheses are closed properly.\n{MANIM_RULES}\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
            
            process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=True)
            
        else:
            safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', command)
            safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
            scene_type = "ThreeDScene" if "3d" in command.lower() else "Scene"
            
            prefill_code = f"from manim import *\nimport numpy as np\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
            
            enhanced_instruction = (
                f"Design and code a highly professional, cinematic Manim educational animation about: '{command}'.\n"
                "You are an expert director and mathematician (like 3Blue1Brown). Think deeply about the concept and intelligently expand on the user's idea by adding necessary mathematical context, beautiful visual elements, and logical step-by-step explanations.\n"
                "Follow these strict design guidelines:\n"
                "1. Professional Layout: Arrange elements elegantly. Use appropriate scaling and positioning so nothing overlaps the screen edges.\n"
                "2. Creative Typography: Be creative and organic with text. Do not restrict yourself to always adding a static title at the top. Place labels and explanations dynamically where they make the most sense visually.\n"
                "3. Color Palette: Use professional Manim colors (BLUE_E, TEAL, YELLOW, RED) to distinguish different components.\n"
                "4. Sequential Animation: Do not show everything at once. Animate step-by-step (e.g., draw axes, plot graph, add labels).\n"
                "5. Flow: Wait for 1 or 2 seconds between major animations using self.wait().\n"
                f"{MANIM_RULES}\n"
                "Write only the python code."
            )
            
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{enhanced_instruction}\n\n### Response:\n```python\n{prefill_code}"
            
            process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=True)

if __name__ == "__main__":
    app()
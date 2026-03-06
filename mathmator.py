import os
import sys

# 🩹 CRITICAL PATCH FOR PYTHON 3.12+ (Fixes manim_voiceover pkg_resources error)
import importlib.metadata
sys.modules['pkg_resources'] = importlib.metadata

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
    "8. TEXT WRAPPING: Use Paragraph('line1', 'line2') for long text.\n"
)

# Optional Voiceover Rule
VOICEOVER_RULE = (
    "9. VOICEOVER (CRITICAL): You are inheriting from VoiceoverScene. You MUST wrap animations in voiceover blocks to explain the math.\n"
    "   Example:\n"
    "   with self.voiceover(text=\"Here we draw a circle.\") as tracker:\n"
    "       self.play(Create(circle), run_time=tracker.duration)\n"
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

has_gpu = check_gpu()

if has_gpu:
    typer.secho("🤖 Mathmator: GPU Detected! 🚀 (Using Base + LoRA with mmap=False to prevent Segfault)", fg=typer.colors.GREEN, bold=True)
    gpu_layers = -1
    use_mmap = False      
    max_tokens_val = 3000 
    ctx_size = 4096
else:
    typer.secho("🤖 Mathmator: CPU Detected! 🐢 (Using Base + LoRA with mmap=True to save RAM)", fg=typer.colors.YELLOW, bold=True)
    gpu_layers = 0
    use_mmap = True       
    max_tokens_val = 2048  
    ctx_size = 4096

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

def generate_storyboard(command: str, use_voiceover: bool) -> str:
    typer.secho("\n🎬 [Director Agent] Planning the cinematic storyboard...", fg=typer.colors.MAGENTA, bold=True)
    
    voice_instruction = "Include a specific, engaging 'Voiceover:' script for each step." if use_voiceover else "Do NOT include voiceover scripts. This is a silent visual."
    
    planner_prompt = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"You are an elite mathematical art director like 3Blue1Brown. The user wants a Manim animation about: '{command}'.\n"
        "Create a brilliant, step-by-step visual storyboard. DO NOT write any Python code.\n"
        "Follow these rules:\n"
        "1. Break the animation into 3 to 4 distinct, highly dynamic visual steps.\n"
        "2. Focus heavily on geometric transformations, moving dots, graphs, and visual metaphors instead of just text.\n"
        "3. NO CLICHES. Do not start with 'Welcome to...'.\n"
        f"4. {voice_instruction}\n\n"
        "### Response:\nStoryboard:\nStep 1:"
    )
    
    try:
        output = llm(
            planner_prompt,
            max_tokens=350,
            stop=["```", "###", "<|end_of_text|>", "<|eot_id|>"],
            temperature=0.6, # إبداع أعلى للمخرج
            echo=False
        )
        storyboard = "Step 1:" + output["choices"][0]["text"].strip()
        typer.secho(f"\n{storyboard}\n", fg=typer.colors.CYAN)
        return storyboard
    except Exception as e:
        typer.secho(f"Director failed: {e}. Falling back to default prompt.", fg=typer.colors.RED)
        return command # Fallback

def process_and_render(prompt_text: str, safe_topic_name: str, quality: Quality, keep_code: bool, prefill: str, use_voice: bool = False, max_retries: int = 2, rules_used: str = MANIM_RULES):
    global LAST_ERROR_LOG
    temp_file = f"{safe_topic_name}_scene.py"
    current_prompt = prompt_text
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                typer.secho(f"\n🔄 [Auto-Healing] Agent is fixing the error... (Attempt {attempt}/{max_retries})", fg=typer.colors.YELLOW, bold=True)
                speak(f"I found an error in the code. Let me fix that for you, attempt {attempt}.", use_voice)
            else:
                typer.secho("\n🧠 Mathmator writing now ...\n", fg=typer.colors.BLUE)
                
            print(prefill, end="", flush=True)

            output = llm(
                current_prompt, 
                max_tokens=max_tokens_val, 
                stop=["<|end_of_text|>", "<|eot_id|>", "###", "```", "class Concept", "\nclass ", "\ndef ", "if __name__", "\n    def ", "\nConceptScene", "\n# Run"], 
                temperature=0.2 if attempt > 0 else 0.1,  
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
            
            manim_code = re.sub(r'\n[ \t]*[a-zA-Z0-9_]+\s*=\s*ConceptScene\(\).*', '', manim_code, flags=re.DOTALL)
            manim_code = re.sub(r'\n[ \t]*scene\.render\(\).*', '', manim_code, flags=re.DOTALL)
            manim_code = re.sub(r'\n[ \t]*if __name__\s*==.*', '', manim_code, flags=re.DOTALL)
            manim_code = re.sub(r'\n[ \t]*ConceptScene\(\).*', '', manim_code, flags=re.DOTALL)
            manim_code = re.sub(r'\n[ \t]*# Run the.*', '', manim_code, flags=re.DOTALL)
            manim_code = re.sub(r'Cone\(\s*radius=', 'Cone(base_radius=', manim_code)

            manim_code = re.sub(r'UpdateFromFunc\s*\(\s*lambda\s+[a-zA-Z0-9_]+\s*:\s*[a-zA-Z0-9_]+\.(.*?)\s*,\s*([a-zA-Z0-9_]+).*?\)', r'\2.animate.\1', manim_code)
            manim_code = re.sub(r'MoveToTarget\s*\(\s*([^,]+)\s*,\s*(\[.*?\])\s*\)', r'\1.animate.move_to(\2)', manim_code)
            def heal_oscillation(match):
                mob_name = match.group(1)
                return f"{mob_name}.add_updater(lambda m, dt: m.shift(UP * np.sin(self.renderer.time) * 0.05))"
            manim_code = re.sub(r'([a-zA-Z0-9_]+)\.begin_oscillation\(.*?\)', heal_oscillation, manim_code)
            manim_code = re.sub(r'([a-zA-Z0-9_]+)\.start_oscillation\(.*?\)', heal_oscillation, manim_code)
            manim_code = re.sub(r'\bCYAN\b', 'TEAL', manim_code)
            manim_code = re.sub(r'\bMAGENTA\b', 'PURPLE', manim_code)
            manim_code = re.sub(r'\bBROWN\b', 'MAROON', manim_code)
            manim_code = re.sub(r'(get_[xyz]_axis_label\([^,]+),\s*color=([^)]+)\)', r'\1).set_color(\2)', manim_code)
            manim_code = re.sub(r'self\.set_background_color\((.*?)\)', r'self.camera.background_color = \1', manim_code)
            # 🛡️ ANTI-CLICHE SHIELD: Silently remove boring default text elements
            manim_code = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*Text\(\s*["\'](Welcome to|In conclusion|This is|A fundamental).*?["\'].*?\)', r'\1 = VGroup()', manim_code, flags=re.IGNORECASE)
            manim_code = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*Paragraph\(.*?\)', r'\1 = VGroup()', manim_code, flags=re.DOTALL)
            
            # Safe Play wrapper that ignores tracker.duration
            def wrap_play(match):
                content = match.group(1)
                if any(keyword in content for keyword in ['Create', 'Write', 'Fade', 'Transform', 'animate', 'Update', 'Move']):
                    return match.group(0)
                return f'self.play(Create({content}))'
            manim_code = re.sub(r'self\.play\((.*?)\)', wrap_play, manim_code)

            manim_code = re.sub(r'(VGroup\s*\(\s*\*?\s*\[\s*)(?:Create|Write|FadeIn|FadeOut)\s*\(\s*([^)]+)\s*\)(\s*for\s*[^\]]+\]\s*\))', r'\1\2\3', manim_code)
            manim_code = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*.*\.set_axis_labels\(.*\)', r'\1 = VGroup() # Auto-healed bad label', manim_code)
            manim_code = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*.*\.set_title\(.*\)', r'\1 = VGroup() # Auto-healed bad title', manim_code)
            manim_code = re.sub(r'([a-zA-Z0-9_]+)\s*=\s*Title\(.*\)', r'\1 = VGroup() # Auto-healed bad Title obj', manim_code)
            manim_code = re.sub(r'(^[ \t]*.*\.set_axis_labels\(.*\))', r'# \1 (Auto-removed)', manim_code, flags=re.MULTILINE)
            manim_code = re.sub(r'(^[ \t]*.*\.set_title\(.*\))', r'# \1 (Auto-removed)', manim_code, flags=re.MULTILINE)

            def scale_long_text(match):
                full_text = match.group(0)
                content = match.group(1)
                if len(content) > 50:
                    return f'{full_text}.scale_to_fit_width(config.frame_width - 1.5)'
                return full_text
            manim_code = re.sub(r'Text\(\s*["\'](.*?)["\']\s*.*?\)', scale_long_text, manim_code)

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

            env = os.environ.copy()
            env["FORCE_COLOR"] = "1" 

            process = subprocess.Popen(
                ["manim", quality_flag, temp_file, "ConceptScene", "-p"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0, 
                env=env
            )
            
            output_bytes = bytearray()
            while True:
                char = process.stdout.read(1)
                if not char and process.poll() is not None:
                    break
                if char:
                    sys.stdout.buffer.write(char) 
                    sys.stdout.buffer.flush()
                    output_bytes.extend(char)     
                    
            if process.returncode != 0:
                error_text = output_bytes.decode('utf-8', errors='replace')
                raise subprocess.CalledProcessError(process.returncode, process.args, output=error_text, stderr=error_text)

            video_path = os.path.join("media", "videos", f"{safe_topic_name}_scene", quality_folder, "ConceptScene.mp4")
            final_video_name = f"{safe_topic_name}.mp4"

            if os.path.exists(video_path):
                shutil.copy(video_path, final_video_name)
                LAST_ERROR_LOG = ""  
                speak("Finished successfully! The video is ready.", use_voice)
                return True 
            else:
                raise FileNotFoundError("Video generated but path not found.")

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_output = getattr(e, 'output', str(e))
            LAST_ERROR_LOG = error_output  
            
            clean_log = re.sub(r'\x1b\[[0-9;]*m', '', error_output)
            short_error = clean_log[-800:].strip()
            
            if attempt == max_retries:
                typer.secho(f"\n❌ Mathmator Agent Failed after {max_retries} attempts.", fg=typer.colors.RED, bold=True)
                typer.secho(error_output, fg=typer.colors.RED)
                explanation = explain_error_speech(error_output)
                speak(explanation, use_voice)
                return False
                
            current_prompt = (
                "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                "CRITICAL ERROR: The python code you just generated crashed with the following error:\n"
                f"```text\n{short_error}\n```\n"
                "Analyze the error carefully. Rewrite the complete, fully functioning code to FIX this error. Ensure all parentheses are closed and objects exist.\n"
                f"{rules_used}\n\n"
                "### Broken Code:\n"
                f"```python\n{manim_code}\n```\n\n"
                f"### Response:\n```python\n{prefill}"
            )
            
        except Exception as e:
            typer.secho(f"An unexpected system error occurred: {e}", fg=typer.colors.RED)
            return False
            
        finally:
            if not keep_code and os.path.exists(temp_file):
                os.remove(temp_file)

def interactive_edit_loop(quality: Quality, keep_code: bool, use_voice: bool = False, voiceover: bool = False):
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
        old_scene_type = match.group(1) if match else "Scene"
        is_3d = "ThreeDScene" in old_scene_type
        
        base_scene = "ThreeDScene" if is_3d else "Scene"
        
        if voiceover:
            scene_type = f"VoiceoverScene, {base_scene}" if is_3d else "VoiceoverScene"
            prefill_code = (
                "from manim import *\n"
                "import numpy as np\n"
                "from manim_voiceover import VoiceoverScene\n"
                "from manim_voiceover.services.gtts import GTTSService\n\n"
                f"class ConceptScene({scene_type}):\n"
                "    def construct(self):\n"
                "        self.set_speech_service(GTTSService(lang='en', tld='com'))\n"
                "        "
            )
            rules = MANIM_RULES + VOICEOVER_RULE
        else:
            scene_type = base_scene
            prefill_code = (
                "from manim import *\n"
                "import numpy as np\n\n"
                f"class ConceptScene({scene_type}):\n"
                "    def construct(self):\n"
                "        "
            )
            rules = MANIM_RULES
            
        error_context = f"\n\nCRITICAL - The previous code crashed with this error:\n{LAST_ERROR_LOG[-400:]}\nPlease fix the error based on this." if LAST_ERROR_LOG else ""
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nMake this code more professional based on the user's edit: {instruction}{error_context}\nEnsure elegant animations and CRITICALLY ensure all Python parentheses are closed properly.\n{rules}\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
        
        process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice, rules_used=rules)

@app.command()
def animate(topic: str = typer.Argument(...),
            quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
            keep_code: bool = typer.Option(False, "--keep-code", "-k"),
            voiceover: bool = typer.Option(False, "--voiceover", "--explain", "-v", help="Enable AI voiceover explanation")):
    
    safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', topic)
    safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
    
    is_3d = "3d" in topic.lower()
    base_scene = "ThreeDScene" if is_3d else "Scene"
    
    if voiceover:
        scene_type = f"VoiceoverScene, {base_scene}" if is_3d else "VoiceoverScene"
        prefill_code = (
            "from manim import *\n"
            "import numpy as np\n"
            "from manim_voiceover import VoiceoverScene\n"
            "from manim_voiceover.services.gtts import GTTSService\n\n"
            f"class ConceptScene({scene_type}):\n"
            "    def construct(self):\n"
            "        self.set_speech_service(GTTSService(lang='en', tld='com'))\n"
            "        "
        )
        rules = MANIM_RULES + VOICEOVER_RULE
    else:
        scene_type = base_scene
        prefill_code = (
            "from manim import *\n"
            "import numpy as np\n\n"
            f"class ConceptScene({scene_type}):\n"
            "    def construct(self):\n"
            "        "
        )
        rules = MANIM_RULES
    
    # 1. Call the Director Agent
    storyboard = generate_storyboard(topic, voiceover)
    
    # 2. Instruct the Programmer Agent
    enhanced_instruction = (
        f"Generate highly professional Manim CE Python code strictly following this storyboard:\n\n{storyboard}\n\n"
        "CRITICAL DIRECTIVES: YOU MUST OBEY THESE RULES OR THE CODE WILL BE REJECTED.\n"
        "1. TRANSLATE PLAN TO CODE: Turn each step of the storyboard into beautiful, sequential Manim code.\n"
        "2. ZERO CLICHES: NO paragraphs of text. NO 'Welcome to' text on screen.\n"
        "3. VISUAL MATHEMATICS ONLY: Your scene MUST consist of Axes, geometric shapes, dynamic lines, dots tracing paths, and MathTex formulas.\n"
        "4. PROHIBIT LONG TEXT: NEVER use `Text()` for sentences. Only use `Text()` for short 1-3 word labels.\n"
        "5. LAYOUT: Prevent overlapping. Use `.to_edge(UP)` or precise coordinates like `move_to([x, y, 0])`.\n"
        f"{rules}\n"
        "Write ONLY the pure Python code."
    )
    
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{enhanced_instruction}\n\n### Response:\n```python\n{prefill_code}"
    
    process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=False, rules_used=rules)
    interactive_edit_loop(quality, keep_code, use_voice=False, voiceover=voiceover)

@app.command()
def edit(instruction: str = typer.Argument(...),
         quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
         keep_code: bool = typer.Option(False, "--keep-code", "-k"),
         voiceover: bool = typer.Option(False, "--voiceover", "--explain", "-v", help="Enable AI voiceover explanation")):
    
    global LAST_ERROR_LOG
    if not os.path.exists(LATEST_CODE_FILE):
        typer.secho("No previous animation found to edit! Run 'animate' first.", fg=typer.colors.RED)
        raise typer.Exit()
        
    with open(LATEST_CODE_FILE, "r", encoding="utf-8") as f:
        current_code = f.read()
        
    safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', instruction)
    safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
    
    match = re.search(r'class ConceptScene\((.*?)\):', current_code)
    old_scene_type = match.group(1) if match else "Scene"
    is_3d = "ThreeDScene" in old_scene_type
    
    base_scene = "ThreeDScene" if is_3d else "Scene"
    
    if voiceover:
        scene_type = f"VoiceoverScene, {base_scene}" if is_3d else "VoiceoverScene"
        prefill_code = (
            "from manim import *\n"
            "import numpy as np\n"
            "from manim_voiceover import VoiceoverScene\n"
            "from manim_voiceover.services.gtts import GTTSService\n\n"
            f"class ConceptScene({scene_type}):\n"
            "    def construct(self):\n"
            "        self.set_speech_service(GTTSService(lang='en', tld='com'))\n"
            "        "
        )
        rules = MANIM_RULES + VOICEOVER_RULE
    else:
        scene_type = base_scene
        prefill_code = (
            "from manim import *\n"
            "import numpy as np\n\n"
            f"class ConceptScene({scene_type}):\n"
            "    def construct(self):\n"
            "        "
        )
        rules = MANIM_RULES
    
    error_context = f"\n\nCRITICAL - The previous code crashed with this error:\n{LAST_ERROR_LOG[-400:]}\nPlease fix the error based on this." if LAST_ERROR_LOG else ""
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nMake this code more professional based on the user's edit: {instruction}{error_context}\nEnsure elegant animations and CRITICALLY ensure all Python parentheses are closed properly.\n{rules}\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
    
    process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=False, rules_used=rules)
    interactive_edit_loop(quality, keep_code, use_voice=False, voiceover=voiceover)

@app.command()
def voice(quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
          keep_code: bool = typer.Option(False, "--keep-code", "-k"),
          voiceover: bool = typer.Option(False, "--voiceover", "--explain", "-v", help="Enable AI voiceover explanation")):
    
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
            old_scene_type = match.group(1) if match else "Scene"
            is_3d = "ThreeDScene" in old_scene_type
            
            base_scene = "ThreeDScene" if is_3d else "Scene"
            
            if voiceover:
                scene_type = f"VoiceoverScene, {base_scene}" if is_3d else "VoiceoverScene"
                prefill_code = (
                    "from manim import *\n"
                    "import numpy as np\n"
                    "from manim_voiceover import VoiceoverScene\n"
                    "from manim_voiceover.services.gtts import GTTSService\n\n"
                    f"class ConceptScene({scene_type}):\n"
                    "    def construct(self):\n"
                    "        self.set_speech_service(GTTSService(lang='en', tld='com'))\n"
                    "        "
                )
                rules = MANIM_RULES + VOICEOVER_RULE
            else:
                scene_type = base_scene
                prefill_code = (
                    "from manim import *\n"
                    "import numpy as np\n\n"
                    f"class ConceptScene({scene_type}):\n"
                    "    def construct(self):\n"
                    "        "
                )
                rules = MANIM_RULES
            
            error_context = f"\n\nCRITICAL - The previous code crashed with this error:\n{LAST_ERROR_LOG[-400:]}\nPlease fix the error based on this." if LAST_ERROR_LOG else ""
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nMake this code more professional based on the user's edit: {command}{error_context}\nEnsure elegant animations and CRITICALLY ensure all Python parentheses are closed properly.\n{rules}\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
            
            process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=True, rules_used=rules)
            
        else:
            safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', command)
            safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
            
            is_3d = "3d" in command.lower()
            base_scene = "ThreeDScene" if is_3d else "Scene"
            
            if voiceover:
                scene_type = f"VoiceoverScene, {base_scene}" if is_3d else "VoiceoverScene"
                prefill_code = (
                    "from manim import *\n"
                    "import numpy as np\n"
                    "from manim_voiceover import VoiceoverScene\n"
                    "from manim_voiceover.services.gtts import GTTSService\n\n"
                    f"class ConceptScene({scene_type}):\n"
                    "    def construct(self):\n"
                    "        self.set_speech_service(GTTSService(lang='en', tld='com'))\n"
                    "        "
                )
                rules = MANIM_RULES + VOICEOVER_RULE
            else:
                scene_type = base_scene
                prefill_code = (
                    "from manim import *\n"
                    "import numpy as np\n\n"
                    f"class ConceptScene({scene_type}):\n"
                    "    def construct(self):\n"
                    "        "
                )
                rules = MANIM_RULES
            
            # 1. Call the Director Agent
            storyboard = generate_storyboard(command, voiceover)
            
            # 2. Instruct the Programmer Agent
            enhanced_instruction = (
                f"Generate highly professional Manim CE Python code strictly following this storyboard:\n\n{storyboard}\n\n"
                "CRITICAL DIRECTIVES: YOU MUST OBEY THESE RULES OR THE CODE WILL BE REJECTED.\n"
                "1. TRANSLATE PLAN TO CODE: Turn each step of the storyboard into beautiful, sequential Manim code.\n"
                "2. ZERO CLICHES: NO paragraphs of text. NO 'Welcome to' text on screen.\n"
                "3. VISUAL MATHEMATICS ONLY: Your scene MUST consist of Axes, geometric shapes, dynamic lines, dots tracing paths, and MathTex formulas.\n"
                "4. PROHIBIT LONG TEXT: NEVER use `Text()` for sentences. Only use `Text()` for short 1-3 word labels.\n"
                "5. LAYOUT: Prevent overlapping. Use `.to_edge(UP)` or precise coordinates like `move_to([x, y, 0])`.\n"
                f"{rules}\n"
                "Write ONLY the pure Python code."
            )
                    
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{enhanced_instruction}\n\n### Response:\n```python\n{prefill_code}"
            
            process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=True, rules_used=rules)

if __name__ == "__main__":
    app()
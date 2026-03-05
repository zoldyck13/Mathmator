import typer
import subprocess
import os
import shutil
import re
import speech_recognition as sr
import sys
import torch
import sounddevice as sd
import zipfile
from enum import Enum
from llama_cpp import Llama

app = typer.Typer()

BASE_MODEL_PATH = "./Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
LORA_PATH = "./mathmator_lora.gguf"
LATEST_CODE_FILE = "latest_mathmator_code.py"

BASE_MODEL_URL = "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
LORA_ZIP_URL = "https://huggingface.co/Alisaadmotar/Mathmator-Llama3-LoRA/resolve/main/mathmator_lora.gguf.zip"
LORA_ZIP_PATH = "./mathmator_lora.gguf.zip"

if not os.path.exists(BASE_MODEL_PATH):
    typer.secho("Base model not found. Downloading Meta-Llama-3-8B-Instruct-Q4_K_M.gguf (~4.7GB)...", fg=typer.colors.YELLOW)
    torch.hub.download_url_to_file(BASE_MODEL_URL, BASE_MODEL_PATH)

if not os.path.exists(LORA_PATH):
    if not os.path.exists(LORA_ZIP_PATH):
        typer.secho("Mathmator LoRA adapter not found. Downloading mathmator_lora.gguf.zip...", fg=typer.colors.YELLOW)
        try:
            torch.hub.download_url_to_file(LORA_ZIP_URL, LORA_ZIP_PATH)
        except Exception as e:
            typer.secho(f"Failed to download LoRA ZIP: {e}", fg=typer.colors.RED)
            sys.exit(1)
            
    typer.secho("Extracting mathmator_lora.gguf...", fg=typer.colors.CYAN)
    try:
        with zipfile.ZipFile(LORA_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        if os.path.exists(LORA_ZIP_PATH):
            os.remove(LORA_ZIP_PATH)
    except Exception as e:
        typer.secho(f"Failed to extract LoRA: {e}", fg=typer.colors.RED)
        sys.exit(1)

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

has_gpu = check_gpu()

if has_gpu:
    typer.secho("🤖 Mathmator: Running on GPU 🚀", fg=typer.colors.GREEN, bold=True)
else:
    typer.secho("🤖 Mathmator: Running on CPU 🐢", fg=typer.colors.YELLOW, bold=True)

with SuppressStderr():
    llm = Llama(
        model_path=BASE_MODEL_PATH, 
        lora_path=LORA_PATH, 
        n_ctx=2048, 
        n_gpu_layers=-1 if has_gpu else 0,
        verbose=False
    )

device = torch.device('cuda' if has_gpu else 'cpu')
model_file = 'silero_v3_en.pt'

if not os.path.isfile(model_file):
    typer.secho("Downloading High-Quality Offline Voice Model (One time only)...", fg=typer.colors.YELLOW)
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt', model_file)

try:
    with SuppressStderr():
        tts_model = torch.package.PackageImporter(model_file).load_pickle("tts_models", "model")
        tts_model.to(device)
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
            typer.secho("⏳ Processing speech with Offline Whisper...", fg=typer.colors.MAGENTA)
            
            command = recognizer.recognize_whisper(audio, model="base.en").strip()
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

def process_and_render(prompt_text: str, safe_topic_name: str, quality: Quality, keep_code: bool, prefill: str, use_voice: bool = False):
    temp_file = f"{safe_topic_name}_scene.py"
    
    try:
        typer.secho("\n🧠 Mathmator writing now ...\n", fg=typer.colors.BLUE)
        print(prefill, end="", flush=True)

        output = llm(
            prompt_text, 
            max_tokens=500,       
            stop=["<|end_of_text|>", "<|eot_id|>", "###", "```", "class Concept", "\nclass ", "\ndef ", "if __name__"], 
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

        manim_code = manim_code.replace("```python", "").replace("```", "").strip()
        manim_code = re.sub(r'ApplyMethod\((.*?)\)', r'\1', manim_code)
        manim_code = re.sub(r'\n[a-zA-Z0-9_]+\s*=\s*ConceptScene\(\).*', '', manim_code, flags=re.DOTALL)
        manim_code = re.sub(r'\nscene\.render\(\).*', '', manim_code, flags=re.DOTALL)
        manim_code = re.sub(r'\nif __name__\s*==.*', '', manim_code, flags=re.DOTALL)
        manim_code = re.sub(r'Cone\(\s*radius=', 'Cone(base_radius=', manim_code)

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

        speak(f"Rendering the video at {quality.value} quality.", use_voice)

        subprocess.run(["manim", quality_flag, temp_file, "ConceptScene", "-p"], check=True)

        video_path = os.path.join("media", "videos", f"{safe_topic_name}_scene", quality_folder, "ConceptScene.mp4")
        final_video_name = f"{safe_topic_name}.mp4"

        if os.path.exists(video_path):
            shutil.copy(video_path, final_video_name)
            speak("Finished successfully! The video is ready.", use_voice)
        else:
            speak("Video was generated but couldn't be found.", use_voice)

    except subprocess.CalledProcessError:
        speak("Something went wrong trying to generate the video.", use_voice)
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
    finally:
        if not keep_code and os.path.exists(temp_file):
            os.remove(temp_file)

def interactive_edit_loop(quality: Quality, keep_code: bool, use_voice: bool = False):
    while True:
        typer.secho("\n" + "="*60, fg=typer.colors.MAGENTA)
        instruction = typer.prompt("✍️  Enter your edit for this video (or type 'exit' to quit)")
        
        if instruction.strip().lower() == "exit":
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
        
        prefill_code = f"from manim import *\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
        
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nModify this Manim code: {instruction}\nCRITICAL: Ensure all Python parentheses and brackets are closed properly.\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
        
        process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice)

@app.command()
def animate(topic: str = typer.Argument(...),
            quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
            keep_code: bool = typer.Option(False, "--keep-code", "-k")):
    
    safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', topic)
    safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
    
    scene_type = "ThreeDScene" if "3d" in topic.lower() else "Scene"
        
    prefill_code = f"from manim import *\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
    
    enhanced_instruction = (
        f"Create a highly professional, cinematic Manim animation for: {topic}. "
        "Follow these design guidelines:\n"
        "1. Professional Layout: Arrange elements elegantly. Use appropriate scaling and positioning so nothing overlaps the screen edges.\n"
        "2. Typography: Always add a clear title at the top of the screen using Text or MathTex.\n"
        "3. Color Palette: Use professional Manim colors (BLUE_E, TEAL, YELLOW, RED) to distinguish different mathematical or logical components.\n"
        "4. Sequential Animation: Do not show everything at once. Animate step-by-step (e.g., Draw axes first, then plot the graph, then add labels).\n"
        "5. Flow: Wait for 1 or 2 seconds between major animations using self.wait().\n"
        "Write only the python code."
    )
    
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{enhanced_instruction}\n\n### Response:\n```python\n{prefill_code}"
    
    process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=False)
    interactive_edit_loop(quality, keep_code, use_voice=False)

@app.command()
def edit(instruction: str = typer.Argument(...),
         quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
         keep_code: bool = typer.Option(False, "--keep-code", "-k")):
    
    if not os.path.exists(LATEST_CODE_FILE):
        typer.secho("No previous animation found to edit! Run 'animate' first.", fg=typer.colors.RED)
        raise typer.Exit()
        
    with open(LATEST_CODE_FILE, "r", encoding="utf-8") as f:
        current_code = f.read()
        
    safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', instruction)
    safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
    
    match = re.search(r'class ConceptScene\((.*?)\):', current_code)
    scene_type = match.group(1) if match else "Scene"
    
    prefill_code = f"from manim import *\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nModify this Manim code: {instruction}\nCRITICAL: Ensure all Python parentheses and brackets are closed properly.\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
    
    process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=False)
    interactive_edit_loop(quality, keep_code, use_voice=False)

@app.command()
def voice(quality: Quality = typer.Option(Quality.low, "--quality", "-q"),
          keep_code: bool = typer.Option(False, "--keep-code", "-k")):
    
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
            
        if "exit" in command or "stop listening" in command or "goodbye" in command:
            speak("Goodbye! Shutting down voice mode.", use_voice=True)
            break
            
        if "animate" in command:
            parts = command.split("animate", 1)
            if len(parts) > 1:
                topic = parts[1].strip()
                speak(f"Got it. Animating {topic}", use_voice=True)
                
                safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', topic)
                safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
                scene_type = "ThreeDScene" if "3d" in topic.lower() else "Scene"
                
                prefill_code = f"from manim import *\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
                
                enhanced_instruction = (
                    f"Create a highly professional, cinematic Manim animation for: {topic}. "
                    "Follow these design guidelines:\n"
                    "1. Professional Layout: Arrange elements elegantly. Use appropriate scaling and positioning so nothing overlaps the screen edges.\n"
                    "2. Typography: Always add a clear title at the top of the screen using Text or MathTex.\n"
                    "3. Color Palette: Use professional Manim colors (BLUE_E, TEAL, YELLOW, RED) to distinguish different mathematical or logical components.\n"
                    "4. Sequential Animation: Do not show everything at once. Animate step-by-step (e.g., Draw axes first, then plot the graph, then add labels).\n"
                    "5. Flow: Wait for 1 or 2 seconds between major animations using self.wait().\n"
                    "Write only the python code."
                )
                
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{enhanced_instruction}\n\n### Response:\n```python\n{prefill_code}"
                
                process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=True)
                
        elif "edit" in command:
            if not os.path.exists(LATEST_CODE_FILE):
                speak("I cannot edit because there is no previous animation.", use_voice=True)
                continue
                
            parts = command.split("edit", 1)
            if len(parts) > 1:
                instruction = parts[1].strip()
                speak(f"Applying edit: {instruction}", use_voice=True)
                
                with open(LATEST_CODE_FILE, "r", encoding="utf-8") as f:
                    current_code = f.read()
                    
                safe_topic_name = re.sub(r'[^a-zA-Z0-9]', '_', instruction)
                safe_topic_name = re.sub(r'_+', '_', safe_topic_name).strip('_')[:50].strip('_')
                
                match = re.search(r'class ConceptScene\((.*?)\):', current_code)
                scene_type = match.group(1) if match else "Scene"
                
                prefill_code = f"from manim import *\n\nclass ConceptScene({scene_type}):\n    def construct(self):\n        "
                prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nModify this Manim code: {instruction}\nCRITICAL: Ensure all Python parentheses and brackets are closed properly.\n\nCurrent Code:\n{current_code}\n\n### Response:\n```python\n{prefill_code}"
                
                process_and_render(prompt, safe_topic_name, quality, keep_code, prefill_code, use_voice=True)

if __name__ == "__main__":
    app()
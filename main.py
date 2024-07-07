from flask import Flask, request, jsonify
from flask_cors import CORS
from pytube import YouTube
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from threading import Thread
from queue import Queue
import os
import glob
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

task_queue = Queue()

def process_task(link, task_id):
    try:
        file_path = None
        # Download audio from YouTube
        if "youtube.com" in link or "youtu.be" in link:
            yt = YouTube(link)
            audio_stream = yt.streams.filter(only_audio=True).first()
            file_path = audio_stream.download(filename=f"{task_id}.mp3")
        # Download audio from Deezer (To be implemented)
        elif "deezer.com" in link:
            # Add Deezer downloading logic
            pass
        # Download audio from Spotify
        elif "spotify.com" in link:
            command = f"spotdl {link} --output ."
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                task_results[task_id] = {"error": result.stderr}
                return

            # Find the downloaded file (assuming it is the only file in the current directory)
            downloaded_files = glob.glob("*.mp3")
            if downloaded_files:
                downloaded_file = downloaded_files[0]
                new_file_path = f"{task_id}.mp3"
                os.rename(downloaded_file, new_file_path)
                file_path = new_file_path
            else:
                task_results[task_id] = {"error": "File not found after download"}
                return
        else:
            task_results[task_id] = {"error": "Unsupported link"}
            return

        if file_path and os.path.isfile(file_path):
            # Process audio file
            result = pipe(file_path, generate_kwargs={"language": "french"})
            os.remove(file_path)  # clean up the downloaded file

            # Extract text and timestamps
            transcription = result["chunks"]
            formatted_result = [{"timestamp": chunk['timestamp'][0], "text": chunk['text']} for chunk in transcription]

            # Store the result
            task_results[task_id] = formatted_result
        else:
            task_results[task_id] = {"error": "File not found after download"}
    except Exception as e:
        task_results[task_id] = {"error": str(e)}

def worker():
    while True:
        task_id, link = task_queue.get()
        process_task(link, task_id)
        task_queue.task_done()

task_results = {}
thread = Thread(target=worker, daemon=True)
thread.start()

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    link = data.get('link')
    if not link:
        return jsonify({"error": "No link provided"}), 400

    task_id = str(len(task_results) + 1)
    task_queue.put((task_id, link))
    return jsonify({"task_id": task_id})

@app.route('/result/<task_id>', methods=['GET'])
def result(task_id):
    result = task_results.get(task_id)
    if not result:
        return jsonify({"error": "Task not found or not completed yet"}), 404
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

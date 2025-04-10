import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile
import uvicorn
import ffmpeg

# GPU detection
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE_TYPE == "cuda" else "float32"

# Load the Faster Whisper model
model = WhisperModel("medium.en", device=DEVICE_TYPE, compute_type=COMPUTE_TYPE)

# Initialize FastAPI app
app = FastAPI()

@app.post("/")
async def transcribe(file: UploadFile = File(...)):
    original_audio_path = "original_audio.tmp"
    converted_audio_path = "temp_audio.wav"

    # Save the uploaded file
    with open(original_audio_path, "wb") as audio_file:
        audio_file.write(await file.read())

    # Convert audio file to 16kHz mono WAV (safe format)
    try:
        ffmpeg.input(original_audio_path).output(
            converted_audio_path, ar=16000, ac=1, format='wav'
        ).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        return {"error": "Audio conversion failed", "details": str(e)}

    # Transcribe the audio using Faster Whisper
    segments, info = model.transcribe(converted_audio_path)

    transcription = " ".join(segment.text for segment in segments).strip()

    # Debug print statements
    print(f"[DEBUG] Detected language: {info.language}, probability: {info.language_probability:.2f}")
    print(f"[DEBUG] Transcription: {transcription}")

    return {"transcription": transcription}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8123)

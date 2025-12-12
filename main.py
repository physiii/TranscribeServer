import os
import io
import struct
import time
import tempfile
import asyncio
from typing import Tuple, Optional

import numpy as np
import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile, Query
import uvicorn
import ffmpeg
import logging

# Logging config (Docker stdout already captured, add timestamps)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("transcribe")

# GPU detection and pinning (require GPU)
GPU_INDEX = int(os.getenv("TRANSCRIBE_GPU_ID", "0"))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available but GPU execution is required.")
torch.cuda.set_device(GPU_INDEX)
DEVICE_TYPE = "cuda"
COMPUTE_TYPE = "float16"

# Load the Faster Whisper model once at startup, pinned to the chosen GPU.
# Using "medium.en" (English-only, faster than large-v3).
logger.info("Loading Whisper model on cuda:%d with compute_type=%s", GPU_INDEX, COMPUTE_TYPE)
model = WhisperModel(
    "medium.en",
    device=DEVICE_TYPE,
    device_index=GPU_INDEX,
    compute_type=COMPUTE_TYPE,
)

app = FastAPI()

# Limit concurrent transcriptions inside this server so we don't overload GPU/CPU
MAX_CONCURRENT_TRANSCRIBE = 4
_transcribe_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIBE)


def _wav_bytes_to_float32(data: bytes) -> Tuple[np.ndarray, int, int]:
    """
    Parse WAV bytes (PCM or IEEE float, incl. extensible) and return
    (audio_float32_mono, sample_rate, n_channels). If channels > 1,
    downmix to mono. Does NOT resample; caller must ensure 16 kHz.
    """
    bio = io.BytesIO(data)
    header = bio.read(12)
    if len(header) != 12:
        raise RuntimeError("Incomplete WAV header")
    riff, _filesize, wave_tag = struct.unpack("<4sI4s", header)
    if riff != b"RIFF" or wave_tag != b"WAVE":
        raise RuntimeError("Not a RIFF/WAVE file")

    fmt_chunk = None
    data_chunk = None

    while True:
        chunk_header = bio.read(8)
        if len(chunk_header) < 8:
            break
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        chunk_payload = bio.read(chunk_size)
        if len(chunk_payload) != chunk_size:
            raise RuntimeError("Truncated WAV chunk")
        if chunk_size % 2 == 1:
            bio.seek(1, io.SEEK_CUR)  # padding byte

        if chunk_id == b"fmt ":
            fmt_chunk = chunk_payload
        elif chunk_id == b"data":
            data_chunk = chunk_payload
        # Ignore other chunks (fact, LIST, etc.)

        if fmt_chunk is not None and data_chunk is not None:
            break

    if fmt_chunk is None or data_chunk is None:
        raise RuntimeError("Missing fmt or data chunk in WAV")
    if len(fmt_chunk) < 16:
        raise RuntimeError("Invalid fmt chunk")

    (
        audio_format,
        n_channels,
        sample_rate,
        _byte_rate,
        _block_align,
        bits_per_sample,
    ) = struct.unpack("<HHIIHH", fmt_chunk[:16])

    # Handle WAVE_FORMAT_EXTENSIBLE
    if audio_format == 0xFFFE and len(fmt_chunk) >= 40:
        cb_size = struct.unpack("<H", fmt_chunk[16:18])[0]
        if cb_size >= 22:
            subformat = fmt_chunk[24:24 + 16]
            audio_format = struct.unpack("<H", subformat[:2])[0]

    little_endian = "<"
    if audio_format == 3:
        if bits_per_sample != 32:
            raise RuntimeError("Unsupported IEEE float width")
        arr = np.frombuffer(data_chunk, dtype=f"{little_endian}f4")
    elif audio_format == 1:
        if bits_per_sample == 8:
            arr = np.frombuffer(data_chunk, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        elif bits_per_sample == 16:
            arr = np.frombuffer(data_chunk, dtype=f"{little_endian}i2").astype(np.float32)
            arr = arr / 32768.0
        elif bits_per_sample == 32:
            arr = np.frombuffer(data_chunk, dtype=f"{little_endian}i4").astype(np.float32)
            arr = arr / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported PCM bit depth: {bits_per_sample}")
    else:
        raise RuntimeError(f"Unsupported WAV audio format: {audio_format}")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)

    return arr.astype(np.float32), sample_rate, n_channels


def _run_ffmpeg_resample(orig_bytes: bytes) -> np.ndarray:
    """
    Convert arbitrary audio bytes to 16 kHz mono WAV via ffmpeg,
    then load into float32 numpy array.
    """
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as f_in:
        f_in.write(orig_bytes)
        in_path = f_in.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        out_path = f_out.name

    try:
        (
            ffmpeg
            .input(in_path)
            .output(out_path, ar=16000, ac=1, format="wav")
            .overwrite_output()
            .run(quiet=True)
        )
        with open(out_path, "rb") as f:
            converted_bytes = f.read()
        arr, framerate, _ = _wav_bytes_to_float32(converted_bytes)

        if framerate != 16000:
            raise RuntimeError(f"ffmpeg resample failed, got sr={framerate}")

        return arr
    finally:
        try:
            os.remove(in_path)
        except OSError:
            pass
        try:
            os.remove(out_path)
        except OSError:
            pass


def _transcribe_audio(
    audio: np.ndarray,
    pipeline: str,
    extra: dict,
    initial_prompt: Optional[str] = None,
    no_speech_threshold: Optional[float] = None,
    word_timestamps: bool = False,
) -> Tuple[str, dict, list]:
    """
    Core transcription helper. Applies our default decoding settings and
    passes optional per-request controls through to faster-whisper.
    """
    kwargs = {
        "language": "en",
        "beam_size": 8,
        "vad_filter": True,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    if no_speech_threshold is not None:
        kwargs["no_speech_threshold"] = no_speech_threshold
    if word_timestamps:
        kwargs["word_timestamps"] = True

    # Materialize the generator so we can both build the full text and
    # optionally return per-segment / per-word details.
    segments, info = model.transcribe(audio, **kwargs)
    segments_list = list(segments)
    transcription = " ".join(seg.text for seg in segments_list).strip()
    audio_seconds = len(audio) / 16000.0 if len(audio) else 0.0
    meta = {
        "pipeline": pipeline,
        "lang": info.language,
        "lang_p": info.language_probability,
        "audio_seconds": audio_seconds,
        **extra,
    }
    logger.debug(
        "[DEBUG] %s: lang=%s p=%.2f audio=%.2fs",
        pipeline,
        info.language,
        info.language_probability,
        audio_seconds,
    )
    return transcription, meta, segments_list


def _process_audio_bytes(
    data: bytes,
    content_type: str,
    filename: str,
    initial_prompt: Optional[str] = None,
    no_speech_threshold: Optional[float] = None,
    word_timestamps: bool = False,
) -> Tuple[str, dict, list]:
    """
    Heavy, blocking path: parse/convert audio and run WhisperModel.transcribe.
    This runs in a worker thread via asyncio.to_thread.
    """

    # 1) Raw PCM path: TWIN can send 16 kHz mono float32 PCM as application/octet-stream
    #    or audio/raw;rate=16000;channels=1.
    ctype = (content_type or "").lower()
    fname = (filename or "").lower()

    start = time.perf_counter()
    meta_extra = {
        "source_ctype": ctype,
        "source_fname": fname,
    }

    if "audio/raw" in ctype or (
        "application/octet-stream" in ctype and fname.endswith(".raw")
    ):
        audio = np.frombuffer(data, dtype=np.float32).astype(np.float32)
        transcription, meta, segments = _transcribe_audio(
            audio,
            pipeline="raw_pcm",
            extra={**meta_extra, "ffmpeg": False},
            initial_prompt=initial_prompt,
            no_speech_threshold=no_speech_threshold,
            word_timestamps=word_timestamps,
        )
        meta["process_time"] = time.perf_counter() - start
        return transcription, meta, segments

    # 2) 16 kHz mono WAV path: TWIN already sends 16 kHz mono WAV -> skip ffmpeg
    if "audio/wav" in ctype or fname.endswith(".wav"):
        try:
            audio, sr, nch = _wav_bytes_to_float32(data)
        except Exception as e:
            logger.warning("Failed to parse WAV directly, falling back to ffmpeg: %s", e)
            audio = None
            sr = None

        if audio is not None and sr == 16000:
            transcription, meta, segments = _transcribe_audio(
                audio,
                pipeline="wav_direct",
                extra={**meta_extra, "channels": nch, "ffmpeg": False},
                initial_prompt=initial_prompt,
                no_speech_threshold=no_speech_threshold,
                word_timestamps=word_timestamps,
            )
            meta["process_time"] = time.perf_counter() - start
            return transcription, meta, segments
        else:
            # Not 16 kHz or failed to parse; fall back to ffmpeg resample
            audio = _run_ffmpeg_resample(data)
            transcription, meta, segments = _transcribe_audio(
                audio,
                pipeline="wav_ffmpeg",
                extra={**meta_extra, "ffmpeg": True},
                initial_prompt=initial_prompt,
                no_speech_threshold=no_speech_threshold,
                word_timestamps=word_timestamps,
            )
            meta["process_time"] = time.perf_counter() - start
            return transcription, meta, segments

    # 3) Other formats (mp3, etc.): use ffmpeg then whisper
    audio = _run_ffmpeg_resample(data)
    transcription, meta, segments = _transcribe_audio(
        audio,
        pipeline="other_ffmpeg",
        extra={**meta_extra, "ffmpeg": True},
        initial_prompt=initial_prompt,
        no_speech_threshold=no_speech_threshold,
        word_timestamps=word_timestamps,
    )
    meta["process_time"] = time.perf_counter() - start
    return transcription, meta, segments


@app.post("/")
async def transcribe(
    file: UploadFile = File(...),
    initial_prompt: Optional[str] = Query(
        default=None,
        description="Optional initial prompt to guide transcription (e.g. names, jargon, topic).",
    ),
    no_speech_threshold: Optional[float] = Query(
        default=None,
        description=(
            "Optional override for faster-whisper's no_speech_threshold. "
            "Higher values make the model more likely to treat segments as silence."
        ),
    ),
    detailed: bool = Query(
        default=False,
        description=(
            "If true, return detailed metadata including segments and timing instead "
            "of just a flat transcription string."
        ),
    ),
    word_timestamps: bool = Query(
        default=False,
        description=(
            "If true, include per-word timestamps and confidence scores (probabilities) "
            "for each segment. Implies detailed output."
        ),
    ),
):
    """
    Transcription endpoint.

    - If client sends 16 kHz mono WAV, we parse it directly and skip ffmpeg.
    - If client sends raw 16 kHz mono float32 PCM (audio/raw or .raw), we feed it directly.
    - Otherwise, we use ffmpeg once per request to convert to 16 kHz mono.
    - Heavy work runs in a worker thread and is bounded by a semaphore for parallelism.
    """
    start_ts = time.perf_counter()
    data = await file.read()
    post_read = time.perf_counter()

    async with _transcribe_semaphore:
        acquired_ts = time.perf_counter()
        transcription, meta, segments = await asyncio.to_thread(
            _process_audio_bytes,
            data,
            file.content_type or "",
            file.filename or "",
            initial_prompt,
            no_speech_threshold,
            bool(word_timestamps),
        )
    end_ts = time.perf_counter()

    total = end_ts - start_ts
    read_time = post_read - start_ts
    wait_time = acquired_ts - post_read
    proc_time = end_ts - acquired_ts
    ctype = file.content_type or "unknown"
    fname = file.filename or "unnamed"
    meta_proc = meta.get("process_time", proc_time)
    pipeline = meta.get("pipeline", "unknown")
    audio_seconds = meta.get("audio_seconds", 0.0)
    ffmpeg_used = meta.get("ffmpeg", False)
    lang = meta.get("lang", "?")
    lang_p = meta.get("lang_p", 0.0)
    text_preview = transcription if len(transcription) <= 160 else transcription[:157] + "..."
    logger.info(
        "Request completed in %.3fs (read=%.3fs wait=%.3fs process=%.3fs/%.3fs, bytes=%d, "
        "ctype=%s, file=%s, pipeline=%s, ffmpeg=%s, audio=%.2fs, lang=%s p=%.2f, text=%r)",
        total,
        read_time,
        wait_time,
        proc_time,
        meta_proc,
        len(data),
        ctype,
        fname,
        pipeline,
        ffmpeg_used,
        audio_seconds,
        lang,
        lang_p,
        text_preview,
    )

    # Backwards-compatible default: simple response with just the transcription.
    if not detailed and not word_timestamps:
        return {"transcription": transcription}

    # Build rich response with metadata, segments, and optional per-word confidences.
    response = {
        "transcription": transcription,
        "language": lang,
        "language_probability": lang_p,
        "audio_seconds": audio_seconds,
        "pipeline": pipeline,
        "ffmpeg_used": ffmpeg_used,
        "timing": {
            "total": total,
            "read": read_time,
            "wait": wait_time,
            "process": proc_time,
            "model": meta_proc,
        },
    }

    segments_out = []
    for seg in segments:
        seg_obj = {
            "id": getattr(seg, "id", None),
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "avg_logprob": getattr(seg, "avg_logprob", None),
            "no_speech_prob": getattr(seg, "no_speech_prob", None),
        }
        if word_timestamps and getattr(seg, "words", None) is not None:
            seg_obj["words"] = [
                {
                    "start": w.start,
                    "end": w.end,
                    "word": w.word,
                    "probability": getattr(w, "probability", None),
                }
                for w in seg.words
            ]
        segments_out.append(seg_obj)

    response["segments"] = segments_out
    return response


if __name__ == "__main__":
    workers = int(os.getenv("TRANSCRIBE_WORKERS", "1"))
    port = int(os.getenv("TRANSCRIBE_PORT", "8123"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=workers,
        loop="uvloop",
        http="httptools",
    )


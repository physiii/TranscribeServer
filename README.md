## TranscribeServer API

This service provides a simple HTTP API for running high-quality speech-to-text transcription using the `faster-whisper` implementation of OpenAI Whisper.

It is implemented as a FastAPI app served by Uvicorn inside Docker.

---

### Model and decoding defaults

- **Model**: `faster-whisper` with the `medium.en` checkpoint (English-only).
- **Device**: GPU if available (`cuda`), otherwise CPU.
- **Sample rate**: audio is converted to **16 kHz mono** before transcription.
- **Decoding defaults** (applied internally on every request):
  - **language**: `"en"` (force English transcription)
  - **beam_size**: `8` (higher quality vs. the default 5, slightly slower)
  - **vad_filter**: `True` (voice activity detection; filters non-speech regions)
  - **temperature**: Whisper/faster-whisper default (`0.0`, deterministic decoding)

These defaults aim for **high accuracy and stability** for English audio.

---

### HTTP API

- **Method**: `POST`
- **Path**: `/`
- **Content type**: `multipart/form-data`
- **Field name**: `file`

The server accepts a single uploaded audio file and returns a JSON object with the transcription.

#### Supported audio formats

- **Preferred**: 16 kHz mono WAV (`audio/wav`)
  - Parsed directly with no extra resampling.
- **Raw PCM**: 16 kHz mono float32 PCM as `application/octet-stream` or `audio/raw` with `.raw` extension.
  - Fed directly to the model.
- **Other formats** (e.g. MP3, other WAV, etc.):
  - Converted once per request using `ffmpeg` to 16 kHz mono WAV.

---

### Request parameters

All parameters are passed as **query parameters**.

#### `initial_prompt` (optional)

- **Type**: string
- **Default**: not set
- **Purpose**: Provide **context / bias** for the transcription.
  - Useful for:
    - Company or product names
    - Speaker names
    - Domain-specific jargon and acronyms
  - The model will try to favor words and style consistent with this prompt.

**Example use cases**:

- "Conversation between Andy and John about Kubernetes deployments and audio transcription."
- "Technical meeting about microservices, Kafka, and PostgreSQL."

The prompt is not returned in the output; it is only used internally to guide decoding.

**Example request (curl)**:

```bash
curl -X POST "http://localhost:8123/?initial_prompt=Andy%20and%20John%20discuss%20Kubernetes%20deployments" \
  -F "file=@/path/to/audio.wav"
```

**Example request (Python)**:

```python
import requests

url = "http://localhost:8123/"
params = {
    "initial_prompt": "Andy and John discussing Kubernetes deployments and audio transcription.",
}
files = {
    "file": ("audio.wav", open("audio.wav", "rb"), "audio/wav"),
}

response = requests.post(url, params=params, files=files, timeout=60)
print(response.json())
```

#### `no_speech_threshold` (optional)

- **Type**: float
- **Default**: `None` (use faster-whisper's internal default)
- **Purpose**: Control how aggressively the model treats segments as **non-speech**.
  - Higher values → **more likely to treat low-confidence segments as silence**.
  - Lower values → more likely to emit text even on borderline or noisy segments.

When provided, this value is passed directly to `WhisperModel.transcribe` as `no_speech_threshold`.
This is useful for tuning behavior on:

- Very quiet recordings
- Short clips
- Environments with lots of background noise

**Example**: Be more conservative, preferring silence over hallucinated text:

```bash
curl -X POST "http://localhost:8123/?no_speech_threshold=0.8" \
  -F "file=@/path/to/audio.wav"
```

You can combine `initial_prompt` and `no_speech_threshold` in the same request:

```bash
curl -X POST "http://localhost:8123/" \
  -G \
  --data-urlencode "initial_prompt=Support call between Andy and a customer about billing" \
  --data-urlencode "no_speech_threshold=0.75" \
  -F "file=@/path/to/support_call.wav"
```

#### `detailed` (optional)

- **Type**: boolean (default: `false`)
- **Purpose**: Control whether the API returns **only the transcription string** (simple mode) or a **richer JSON object** with metadata and segment information.
  - When `false` (default), the response is:
    - `{"transcription": "<full text>"}`
  - When `true`, the response includes:
    - language, language probability
    - audio duration and pipeline information
    - timing breakdown
    - per-segment information (start/end/time/text, etc.)

This is useful when clients want more insight into how the model decoded the audio, but do not need per-word details.

#### `word_timestamps` (optional)

- **Type**: boolean (default: `false`)
- **Purpose**: If `true`, the server will:
  - Ask `faster-whisper` for **word-level timestamps** (`word_timestamps=True`)
  - Return **per-word start/end times and confidence scores** (probabilities) inside each segment.
- **Behavior**:
  - Setting `word_timestamps=true` **implicitly enables detailed mode**, so the response will be the rich JSON object described below.

This is the flag to use if you want per-word confidences for alignment, highlighting, or karaoke-style playback.

---

### Response schema

On success, the API returns one of two shapes depending on the query parameters:

#### Simple response (default)

```json
{
  "transcription": "Thank you for watching."
}
```

- **`transcription`**: the full transcribed text for the provided audio.

This is the default when **both** `detailed=false` and `word_timestamps=false` (or when they are omitted).

#### Detailed response (when `detailed=true` or `word_timestamps=true`)

When `detailed=true` (or `word_timestamps=true`), the response includes extra metadata and a list of segments. Example:

```json
{
  "transcription": "Thank you for watching.",
  "language": "en",
  "language_probability": 0.99,
  "audio_seconds": 3.5,
  "pipeline": "wav_direct",
  "ffmpeg_used": false,
  "timing": {
    "total": 0.120,
    "read": 0.005,
    "wait": 0.000,
    "process": 0.115,
    "model": 0.110
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.2,
      "text": "Thank you",
      "avg_logprob": -0.05,
      "no_speech_prob": 0.01,
      "words": [
        {
          "start": 0.0,
          "end": 0.5,
          "word": "Thank",
          "probability": 0.98
        },
        {
          "start": 0.5,
          "end": 1.2,
          "word": "you",
          "probability": 0.97
        }
      ]
    },
    {
      "id": 1,
      "start": 1.2,
      "end": 3.5,
      "text": "for watching.",
      "avg_logprob": -0.03,
      "no_speech_prob": 0.02,
      "words": [
        {
          "start": 1.2,
          "end": 2.0,
          "word": "for",
          "probability": 0.99
        },
        {
          "start": 2.0,
          "end": 3.5,
          "word": "watching.",
          "probability": 0.98
        }
      ]
    }
  ]
}
```

- **`language`** / **`language_probability`**: detected language and its probability from `faster-whisper`.
- **`audio_seconds`**: duration of the (resampled) audio in seconds.
- **`pipeline`**: which internal path was used (`raw_pcm`, `wav_direct`, `wav_ffmpeg`, or `other_ffmpeg`).
- **`ffmpeg_used`**: whether ffmpeg was invoked for resampling/format conversion.
- **`timing`**: high-level timing breakdown for the request.
- **`segments`**: list of decoded segments:
  - `id`, `start`, `end`, `text`, `avg_logprob`, `no_speech_prob`.
  - If `word_timestamps=true`:
    - Each segment includes `words`, a list of:
      - `start`, `end`: word-level timestamps (seconds).
      - `word`: the token/word as a string.
      - `probability`: confidence-like score from `faster-whisper`.

If `detailed=true` but `word_timestamps=false`, the schema is the same except that `words` is omitted from each segment.

Any server-side logging (language detection, processing time, etc.) is written to stdout (Docker logs) and not included in the HTTP response.

---

### Performance and behavior notes

- Using `large-v3` with beam size 8 and VAD on a modern GPU (e.g. RTX 4090) can transcribe **several times faster than real time** for typical clip lengths.
- For very short, nearly silent clips, you may want to tune `no_speech_threshold` upwards (e.g. `0.75–0.85`) to reduce hallucinated phrases like generic closings.
- For domain-heavy content, providing a good `initial_prompt` can significantly improve recognition of names, acronyms, and technical terms.

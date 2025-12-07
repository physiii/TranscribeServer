## TranscribeServer API

This service provides a simple HTTP API for running high-quality speech-to-text transcription using the `faster-whisper` implementation of OpenAI Whisper.

It is implemented as a FastAPI app served by Uvicorn inside Docker.

---

### Model and decoding defaults

- **Model**: `faster-whisper` with the `large-v3` checkpoint.
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

---

### Response schema

On success, the API returns:

```json
{
  "transcription": "Thank you for watching."
}
```

- **`transcription`**: the full transcribed text for the provided audio.

Any server-side logging (language detection, processing time, etc.) is written to stdout (Docker logs) and not included in the HTTP response.

---

### Performance and behavior notes

- Using `large-v3` with beam size 8 and VAD on a modern GPU (e.g. RTX 4090) can transcribe **several times faster than real time** for typical clip lengths.
- For very short, nearly silent clips, you may want to tune `no_speech_threshold` upwards (e.g. `0.75–0.85`) to reduce hallucinated phrases like generic closings.
- For domain-heavy content, providing a good `initial_prompt` can significantly improve recognition of names, acronyms, and technical terms.

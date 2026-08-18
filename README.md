---
title: Speech To Text API
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🎙️ Speech-to-Text API

A from-scratch Automatic Speech Recognition REST API built with a
**CNN + Bidirectional GRU + CTC** architecture, trained on the LibriSpeech
corpus. No pretrained models were used — every component (feature
extraction, CNN, BiGRU, CTC loss, greedy decoder) was implemented and
trained from scratch.

## Architecture

```
Raw Audio (.wav/.flac)
      ↓
Log-Mel Spectrogram (128 mel bins)
      ↓
CNN (stride=2) + ResidualCNN x3
      ↓
Bidirectional GRU x5 (hidden=384)
      ↓
Linear Classifier + CTC
      ↓
Greedy Decoder → Transcript
```

## Training Details

| | |
|---|---|
| Dataset | LibriSpeech `dev-clean` (~5.4h) |
| Framework | PyTorch |
| Parameters | ~21.9M |
| Val WER | 0.597 |
| Val CER | 0.234 |

## API Usage

### Health check
```bash
curl https://<your-space-url>/health
```

### Transcribe an audio file
```bash
curl -X POST "https://<your-space-url>/transcribe" \
  -F "file=@sample.wav"
```

**Response:**
```json
{
  "transcript": "she has a son theft and a daughter hunger",
  "duration_sec": 4.32,
  "inference_sec": 0.87,
  "real_time_factor": 0.20,
  "word_count": 8
}
```

### Interactive docs
Visit `/docs` for the auto-generated Swagger UI to test the API directly
in your browser.

## Tech Stack

- **FastAPI** — async REST API framework
- **PyTorch / torchaudio** — model and audio processing
- **Docker** — containerized deployment
- **Hugging Face Spaces** — hosting

## Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

Then open `http://localhost:7860/docs`.

## Limitations

Trained on only 5.4 hours of audio (vs.\ the 100h+ typical for production
ASR systems) with no language model. This is a research/learning project
demonstrating the full ASR pipeline built from first principles — not a
production-grade transcription service.

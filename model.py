"""
Model definition, feature extraction, and inference logic for the
CNN + BiGRU + CTC speech-to-text model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS      = 128
N_FFT       = 512
HOP_LENGTH  = 160
WIN_LENGTH  = 400


# ─────────────────────────────────────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
class TextTransform:
    def __init__(self):
        chars = ['<blank>'] + list('abcdefghijklmnopqrstuvwxyz') + [' ', "'"]
        self.char2idx  = {c: i for i, c in enumerate(chars)}
        self.idx2char  = {i: c for c, i in self.char2idx.items()}
        self.blank_idx = 0

    def int_to_text(self, indices):
        return ''.join(
            self.idx2char.get(i, '') for i in indices if i != self.blank_idx
        )

    def __len__(self):
        return len(self.char2idx)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO TRANSFORM
# ─────────────────────────────────────────────────────────────────────────────
class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
        )

    def forward(self, x):
        x = self.mel(x)
        x = torch.log(x + 1e-9)
        x = (x - x.mean()) / (x.std() + 1e-9)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, n_feats, dropout):
        super().__init__()
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.cnn1        = nn.Conv2d(in_channels, out_channels, kernel,
                                     stride=stride, padding=kernel // 2)
        self.dropout1    = nn.Dropout(dropout)
        self.layer_norm2 = CNNLayerNorm(n_feats)
        self.cnn2        = nn.Conv2d(out_channels, out_channels, kernel,
                                     stride=stride, padding=kernel // 2)
        self.dropout2    = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x); x = F.gelu(x)
        x = self.dropout1(x);    x = self.cnn1(x)
        x = self.layer_norm2(x); x = F.gelu(x)
        x = self.dropout2(x);    x = self.cnn2(x)
        return x + residual


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first=False):
        super().__init__()
        self.BiGRU      = nn.GRU(rnn_dim, hidden_size, num_layers=1,
                                 batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x); x = F.gelu(x)
        x, _ = self.BiGRU(x);   x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim,
                 n_class, n_feats, stride=2, dropout=0.1):
        super().__init__()
        self.cnn = nn.Conv2d(1, 32, kernel_size=3, stride=stride, padding=1)
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, 3, 1, n_feats, dropout + i * 0.05)
            for i in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(32 * n_feats, rnn_dim)
        self.birnn_layers = nn.Sequential(
            BidirectionalGRU(rnn_dim,     rnn_dim, dropout, batch_first=True),
            *[BidirectionalGRU(rnn_dim*2, rnn_dim, dropout, batch_first=True)
              for _ in range(n_rnn_layers - 1)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),    nn.GELU(), nn.Dropout(dropout),
            nn.Linear(rnn_dim,     rnn_dim//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(rnn_dim//2,  n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        B, C, freq, T = x.size()
        x = x.view(B, C * freq, T).transpose(1, 2)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=2)


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY DECODER
# ─────────────────────────────────────────────────────────────────────────────
def greedy_decode(log_probs, text_transform):
    pred_indices = log_probs.argmax(dim=-1).squeeze(0).cpu().tolist()
    blank_idx    = text_transform.blank_idx
    collapsed    = []
    prev         = None
    for idx in pred_indices:
        if idx != blank_idx and idx != prev:
            collapsed.append(idx)
        prev = idx
    return text_transform.int_to_text(collapsed)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL WRAPPER — loaded once at API startup
# ─────────────────────────────────────────────────────────────────────────────
class SpeechToTextEngine:
    """
    Loads the model once and exposes a single `.transcribe()` method.
    Instantiated once in main.py at FastAPI startup (not per-request).
    """
    def __init__(self, checkpoint_path: str):
        self.device          = torch.device('cpu')
        self.text_transform  = TextTransform()
        self.audio_transform = LogMelSpectrogram()

        ckpt    = torch.load(checkpoint_path, map_location=self.device,
                             weights_only=False)
        hparams = ckpt['hparams']

        self.model = SpeechRecognitionModel(
            hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
            hparams['n_class'],      hparams['n_feats'],      hparams['stride'],
            hparams['dropout']
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        self.epoch   = ckpt.get('epoch')
        self.val_wer = ckpt.get('val_wer')
        self.val_cer = ckpt.get('val_cer')
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def transcribe(self, audio_path: str) -> dict:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

        spec = self.audio_transform(waveform)
        spec = (spec - spec.mean()) / (spec.std() + 1e-9)
        spec = spec.unsqueeze(0).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            log_probs = self.model(spec)
        inference_time = time.time() - t0

        transcript = greedy_decode(log_probs, self.text_transform)
        duration   = waveform.shape[1] / SAMPLE_RATE
        rtf        = inference_time / duration if duration > 0 else 0.0

        return {
            "transcript":       transcript,
            "duration_sec":     round(duration, 3),
            "inference_sec":    round(inference_time, 3),
            "real_time_factor": round(rtf, 3),
            "word_count":       len(transcript.split()) if transcript.strip() else 0,
        }

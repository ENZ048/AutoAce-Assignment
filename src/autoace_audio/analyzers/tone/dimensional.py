"""Arm B: audeering wav2vec2 dimensional SER (arousal/dominance/valence in [0,1])
+ deterministic valence-arousal region mapping. Zero marginal cost, fully local.
Known limits (memo): English-tuned; hears agent+customer mixed."""

from dataclasses import dataclass

import numpy as np

from autoace_audio.analyzers.tone.base import ToneResult
from autoace_audio.analyzers.vad import VadMap
from autoace_audio.config import get_settings
from autoace_audio.schema import EmotionalIntensity, EmotionalTone

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
CHUNK_S = 20.0


def map_va(arousal: float, valence: float) -> tuple[EmotionalTone, EmotionalIntensity]:
    s = get_settings()
    if valence >= s.va_satisfied_v:
        tone = EmotionalTone.SATISFIED
    elif valence < s.va_distressed_v and arousal >= s.va_distressed_a:
        tone = EmotionalTone.DISTRESSED
    elif valence < s.va_upset_v and arousal >= s.va_upset_a:
        tone = EmotionalTone.UPSET
    elif valence < s.va_frustrated_v and arousal >= s.va_frustrated_a_min:
        tone = EmotionalTone.FRUSTRATED
    else:
        tone = EmotionalTone.NEUTRAL
    if arousal < s.intensity_a_low:
        intensity = EmotionalIntensity.LOW
    elif arousal <= s.intensity_a_high:
        intensity = EmotionalIntensity.MEDIUM
    else:
        intensity = EmotionalIntensity.HIGH
    return tone, intensity


def confidence_from_va(arousal: float, valence: float) -> float:
    """Confidence heuristic: distance to the nearest map_va() region boundary.
    Reaches dim_confidence_floor exactly ON a boundary (boundary_dist == 0 --
    maximum ambiguity between two adjacent V-A regions) and rises linearly
    toward dim_confidence_ceiling the further arousal/valence sit from every
    boundary."""
    s = get_settings()
    boundary_dist = min(
        abs(valence - s.va_satisfied_v),
        abs(valence - s.va_upset_v),
        abs(valence - s.va_frustrated_v),
        abs(arousal - s.va_upset_a),
    )
    return float(
        np.clip(
            s.dim_confidence_floor + s.dim_confidence_slope * boundary_dist,
            s.dim_confidence_floor,
            s.dim_confidence_ceiling,
        )
    )


@dataclass
class _Lazy:
    processor: object | None = None
    model: object | None = None


_L = _Lazy()


def _load():
    if _L.model is None:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features):
                import torch as t

                x = self.dropout(features)
                x = t.tanh(self.dense(x))
                x = self.dropout(x)
                return self.out_proj(x)

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.classifier(hidden)

        _L.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        _L.model = EmotionModel.from_pretrained(MODEL_ID)
        _L.model.eval()
        torch.set_grad_enabled(False)
    return _L.processor, _L.model


def _avd(samples: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Duration-weighted mean (arousal, dominance, valence) over 20s chunks."""
    processor, model = _load()
    chunk = int(CHUNK_S * sr)
    outs, weights = [], []
    for i in range(0, samples.size, chunk):
        x = samples[i : i + chunk]
        if x.size < sr:  # skip sub-second tails
            continue
        inputs = processor(x, sampling_rate=sr, return_tensors="pt")
        y = model(inputs.input_values)[0].numpy()
        outs.append(y)
        weights.append(x.size)
    if not outs:
        return 0.5, 0.5, 0.5
    m = np.average(np.stack(outs), axis=0, weights=weights)
    return float(m[0]), float(m[1]), float(m[2])


def classify(samples: np.ndarray, sr: int, vad: VadMap) -> ToneResult:
    from autoace_audio.analyzers.noise import slice_segments  # speech-only audio

    speech = slice_segments(samples, sr, vad.speech)
    if speech.size < sr:
        speech = samples
    arousal, dominance, valence = _avd(speech, sr)
    tone, intensity = map_va(arousal, valence)
    confidence = confidence_from_va(arousal, valence)
    return ToneResult(
        tone=tone,
        intensity=intensity,
        confidence=confidence,
        raw={"arousal": arousal, "dominance": dominance, "valence": valence},
    )

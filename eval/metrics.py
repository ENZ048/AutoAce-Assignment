"""Self-contained metrics (no sklearn dependency)."""

from collections import defaultdict


def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    m: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred, strict=True):
        m[t][p] += 1
    return {t: dict(row) for t, row in m.items()}


def _prf(y_true: list[str], y_pred: list[str], cls: str) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    classes = sorted(set(y_true))
    return sum(_prf(y_true, y_pred, c) for c in classes) / len(classes)


def field_report(labels: dict[str, dict], preds: dict[str, dict]) -> str:
    """Markdown report: per-field accuracy; macro F1 + confusion for emotional_tone."""
    names = [n for n in labels if n in preds]
    lines = [f"# Evaluation report ({len(names)} clips)", ""]
    for f in [
        "emotional_tone",
        "emotional_intensity",
        "background_noise_present",
        "background_noise_severity",
        "audio_quality",
        "speaker_overlap_present",
        "long_silence_present",
    ]:
        pairs = [(str(labels[n][f]), str(preds[n][f])) for n in names if f in labels[n]]
        if not pairs:
            continue
        acc = sum(t == p for t, p in pairs) / len(pairs)
        lines.append(f"- **{f}**: accuracy {acc:.0%} ({len(pairs)} clips)")
    tones = [
        (labels[n]["emotional_tone"], preds[n]["emotional_tone"])
        for n in names
        if "emotional_tone" in labels[n]
    ]
    if not tones:
        # No clip in this batch carries emotional_tone truth at all (e.g. a
        # validation_manifest.csv run where every row is a single-field synthetic
        # augmentation) -- macro_f1/confusion need at least one class to divide by,
        # so skip the tone section rather than crash on an empty class set.
        lines += ["", "**emotional_tone: no labeled clips in this batch**"]
        return "\n".join(lines)
    y_t, y_p = [t for t, _ in tones], [p for _, p in tones]
    lines += ["", f"**emotional_tone macro F1: {macro_f1(y_t, y_p):.3f}**", "", "## Tone confusion"]
    conf = confusion(y_t, y_p)
    classes = sorted(set(y_t) | set(y_p))
    lines.append("| true\\pred | " + " | ".join(classes) + " |")
    lines.append("|---" * (len(classes) + 1) + "|")
    for t in classes:
        lines.append(
            f"| {t} | " + " | ".join(str(conf.get(t, {}).get(p, 0)) for p in classes) + " |"
        )
    return "\n".join(lines)

# AutoAce Call-Audio Analysis

Analyzes production call audio into a structured 9-field JSON: emotional tone,
background noise, technical quality, speaker overlap, long silences, confidence.

## Quickstart

    make setup
    cp .env.example .env   # add GEMINI_API_KEY (paid tier)
    make analyze DIR=data/

Full architecture: `docs/superpowers/specs/2026-07-16-autoace-backend-design.md`.

# orca-clap
Contrastive Language-Audio Pretraining (CLAP) model that recognizes similarities between audio-text pairs, ingesting Orcasound community reports and Orcahello moderator annotations as training data. Primary applications include natural language search of audio files, and annotation assistance.

Proposed layout:
- python/ (inference server—tiny FastAPI with /embed, /score, /nearest)
- tools/
  - node-text-audio-pairs/ (Node utility for generating pairs)
  -   shared CSV schema samples
- models/ (download/readme scripts, no weights checked in)
- docs/ (usage + examples)

APIs this repo should expose:
- POST /score → { audio_url | wav:bytes, prompts: [text...] } -> { scores: [{prompt, sim}], version }
- POST /nearest → { audio_url, k } -> { neighbors: [{audio_id, sim, meta}], version }
- POST /caption (optional/MVP later) → { audio_url } -> { caption, evidence_neighbors }

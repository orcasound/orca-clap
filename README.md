# orca-clap
Contrastive Language-Audio Pretraining (CLAP) model that recognizes similarities between audio-text pairs, ingesting Orcasound community reports and Orcahello moderator annotations as training data. Primary applications include natural language search of audio files, and annotation assistance.

Proposed layout:
- python/ (inference server—tiny FastAPI with /embed, /score, /nearest)
- tools/
  - node-text-audio-pairs/ (Node utility for generating pairs)
  -   shared CSV schema samples
- models/ (download/readme scripts, no weights checked in)
- docs/ (usage + examples)

# Train machine learning models for the endpoints

## Audio to Text adapter


Training:
```
python -u audio_to_text_adapter.py  data_path=/path/to/orcasound_positive_with_embeddings.parquet devices=0 strategy="auto" batch_size=64 learning_rate=0.001 num_workers=5 max_epochs=100

```

Inference:
```
# not implemented
```

## Auto Audio Captioning

Training:
```
python -u create_models/audio_captioning_adapter.py mode="train" whisper_embedding_file="data/whisper_embeddings.csv" decoder_model="" outputs="outputs/"
```

Inference:
```
python -u create_models/audio_captioning_adapter.py mode="infer" files_to_infer="/path/to/files/orcasound/*.flac" decoder_model="" outputs="outputs/" adapter_path="outputs/embedding_model-deepseekr1-7b.pth"
```






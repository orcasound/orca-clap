from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API settings
    api_title: str = "Orcasound Combined Text and Audio Retrieval Service"
    api_version: str = "1.0.0"

    # Text embedding settings
    text_model_name: str = "BAAI/bge-m3"
    text_dataset_path: str = "hf://datasets/bnestor/orcasound_audio_as_text/orcasound_audio_to_text_embeddings.parquet"
    
    # Audio embedding settings
    audio_model_name: str = "openai/whisper-tiny"
    audio_dataset_path: str = "hf://datasets/bnestor/orcasound_whisper_embeddings/orcasound_embeddings_whisper-tiny.parquet"

    

    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()


keep_columns = [ 'audioUri', 'average confidence', 'bucket', 'comments', 'confidence', 'embeddings_list', 'id', 'interestLabel', 'location/latitude', 'location/longitude', 'location/name', 'locationName', 'moderated', 'moderator', 'spectrogramUri', 'state', 'timestamp']
keep_columns = ["audioUri","timestamp","embeddings_list"]
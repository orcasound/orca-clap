"""
audio_to_text_adapter.py

Train an adapter that converts audio embeddings to text embeddings
"""

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from safetensors.torch import save_file, load_file

import hydra
from omegaconf import DictConfig, OmegaConf

import os
import random
from tqdm import tqdm
from codecarbon import EmissionsTracker




def sample_resolve_nan(x: pd.Series,):
    """
    randomly sample a column that is not nan from the eligible annotation columns
    """

    # old dfs had a "state" column
    return_string = ""
    if "state" in x.index:
        if x["state"].lower() =="positive":
            return_string = "Southern Resident Killer Whales. "


        eligible_columns = ["comments",]

        options = x[eligible_columns].values
        options = [o for o in options if isinstance(o, str)]

        return_string = return_string + ". ".join(options).strip()


        tags = [col for col in x.index if col.startswith("tags/")]

        options = x[tags].values
        options = [o for o in options if isinstance(o, str)]

        if len(options)>0:
            if return_string =="":
                return_string = "TAGS:" + ", ".join(options)
            else:
                return_string = ". ".join([return_string.strip('.'), "TAGS:" + ", ".join(options)])

        return return_string

    # new dfs just have a description column
    if "description" in x.index:
        return_string = x["description"].strip()
        return return_string

def get_text_description(label):
    if label.lower() == "oo":
        options = ["A Southern Resident Killer Whale", "The SRKWs","an SRKW", "resident killer whales", " A Killer Whale","Orcinus Orca","Orca","A recording of Orcas","Orca calls and clicks"]
        return random.choice(options)
    elif label.lower() =="noise":
        options = ["General ocean noises","Tidal noises", "Ocean background noise","Hydrophone audio", "An ocean audio recording"]
        return random.choice(options)

class OrcaHelloAdapterDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_df, tokenizer):
        self.embeddings_df = embeddings_df
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.embeddings_df)
    def __getitem__(self, idx):
        embeddings = self.embeddings_df.iloc[idx]["embeddings_list"]
        text = self.embeddings_df.iloc[idx]["string_to_embed"]

        if isinstance(embeddings, list):
            embeddings = [float(i) for i in embeddings]
        elif isinstance(embeddings, str):
            embeddings = embeddings.strip("[").strip("]")
            embeddings = embeddings.split(" ")
            embeddings = [float(i.strip()) for i in embeddings if i.strip()!=""]
        else:
            pass

        on_target = torch.randint(0,2,(1,)) # optional augmentation
        if on_target ==0:
            # randomly sample another string from dataframe
            random_index = random.randint(0,len(self.embeddings_df)-1)
            text = self.embeddings_df.iloc[random_index]["string_to_embed"]
            on_target = on_target -1 # Dissimilar


        tokenized_text = self.tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt") # pad
        tokenized_text = {k:v.squeeze(0) for k,v in tokenized_text.items()} # remove batch dim
        tokenized_text.update({"embedding":torch.tensor(embeddings, dtype=torch.float32), "target":on_target})



        return tokenized_text
    

class AudioToTextAdapter(torch.nn.Module):
    def __init__(self, audio_dim=768, text_dim=768, bottleneck=128):
        super().__init__()
        self.norm = torch.nn.LayerNorm(audio_dim)
        self.fc1 = torch.nn.Linear(audio_dim, bottleneck)
        self.fc2 = torch.nn.Linear(bottleneck, text_dim)

    def forward(self, audio_feat):
        # audio_feat: (batch, audio_dim)
        x = self.norm(audio_feat)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)               # (batch, text_dim)
        out = F.normalize(out, dim=-1)  # L2‑norm for cosine loss
        return out
    


class AudioToTextLightningModule(LightningModule):
    def __init__(self, adapter, text_model, tokenizer, learning_rate=0.005):
        super().__init__()
        self.adapter = adapter
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CosineEmbeddingLoss()
    
    def forward(self, audio_feat):
        return self.adapter(audio_feat)
    
    def training_step(self, batch, batch_idx):
        encoded_data = batch.pop("embedding")
        target = batch.pop("target")
        with torch.no_grad():
            embedded_text = self.text_model(**batch)
        
        logits = self.adapter(encoded_data)
        loss = self.criterion(logits, embedded_text.pooler_output, target.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        encoded_data = batch.pop("embedding")
        target = batch.pop("target")
        with torch.no_grad():
            embedded_text = self.text_model(**batch)
        
        logits = self.adapter(encoded_data)
        loss = self.criterion(logits, embedded_text.pooler_output, target.squeeze())
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.learning_rate)
        return optimizer

    



@hydra.main(version_base=None, config_path="configs", config_name="audio_to_text_adapter")
def main(cfg: DictConfig):
    """
    check if training or inference
    """
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "inference":
        raise NotImplementedError
        inference(cfg)

        
def train(cfg: DictConfig):
    """
    Train or infer using the model
    """
  
    device = "cuda" if torch.cuda.is_available() else "cpu"


    #BAAI/bge-m3
    text_model = AutoModel.from_pretrained(cfg.text_model_name, cache_dir=cfg.hf_cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name, cache_dir=cfg.hf_cache_dir)

    # Adapter = torch.nn.Linear(len(df.iloc[0]["embeddings_list"]), text_model.config.hidden_size)
    Adapter = AudioToTextAdapter(audio_dim = cfg.audio_dim, text_dim=text_model.config.hidden_size)


    # ptl model
    ptl_model = AudioToTextLightningModule(Adapter, text_model, tokenizer, learning_rate=cfg.learning_rate)

    # load the dataset
    orcahello_df = pd.read_parquet(cfg.data_path)
    if "description" in orcahello_df.columns:
        orcahello_df = orcahello_df.loc[orcahello_df["description"].notna(), :]

    strings_to_embed = []
    for i, row in orcahello_df.iterrows():
        # print(row)
        out = sample_resolve_nan(row)
        if i< 10:
            print(out)
        strings_to_embed.append(out)

    orcahello_df.loc[:,"string_to_embed"] = strings_to_embed
    # split into train and val:
    orcahello_df.loc[:, 'timestamp'] = pd.to_datetime(orcahello_df['timestamp'], errors='raise', format='%Y-%m-%dT%H:%M:%S.%fZ')

    # print(max_date, min_date)
    print(orcahello_df['timestamp'].min(), orcahello_df['timestamp'].max())



    # split by time
    train_df = orcahello_df[orcahello_df['timestamp'] < pd.Timestamp(cfg.validation_start_date)]
    val_df = orcahello_df[orcahello_df['timestamp'] >= pd.Timestamp(cfg.validation_start_date)]

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    training_dataset = OrcaHelloAdapterDataset(train_df, tokenizer)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, shuffle=True, num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    validation_dataset = OrcaHelloAdapterDataset(val_df, tokenizer)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, num_workers=cfg.num_workers, batch_size=cfg.batch_size)



    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        # devices=cfg.devices, # not valid on cpu
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        strategy=cfg.strategy,
        precision=cfg.precision,
        log_every_n_steps=5,
        callbacks=[
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-checkpoint"),
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
    )

    with EmissionsTracker(measure_power_secs=60) as tracker:
        trainer.fit(ptl_model, training_dataloader, validation_dataloader)

    # load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    ptl_model = AudioToTextLightningModule.load_from_checkpoint(best_model_path, adapter=Adapter, text_model=text_model, tokenizer=tokenizer)


    # load best model
    best_model_path = trainer.checkpoint_callback.best_model_path

    if os.path.isdir(best_model_path):
        # it is a deepspeed folder, convert it to the regular checkpoint
        from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
        convert_zero_checkpoint_to_fp32_state_dict(best_model_path, os.path.join(best_model_path, "best-checkpoint-final.ckpt"))
        best_model_path = os.path.join(best_model_path, "best-checkpoint-final.ckpt")

    print(f"Best model path: {best_model_path}")

    try:
        ptl_model = AudioToTextLightningModule.load_from_checkpoint(best_model_path, text_model=text_model, adapter=Adapter, tokenizer=tokenizer, learning_rate=cfg.learning_rate)
    except Exception:
        # getting error missing keys "text_model.model.embed_tokens.weight"
        print(sorted(ptl_model.state_dict().keys())[:15])
        state_dict = load_file(best_model_path)
        print(sorted(state_dict.keys())[:15])
        ptl_model.load_state_dict(state_dict, strict=True)


    # adapter = ptl_model.adapter



    torch.save(ptl_model.adapter.state_dict(), os.path.join(cfg.output_path, f"adapter-{cfg.text_model_name.replace('/', '-')}.pth"))
    #safetensors
    save_file(ptl_model.adapter.state_dict(), os.path.join(cfg.output_path, f"adapter-{cfg.text_model_name.replace('/', '-')}.safetensors"))




if __name__ == "__main__":
  main()
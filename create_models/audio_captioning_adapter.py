"""
audio_captioning_adapter.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

import pandas as pd
import random

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from safetensors.torch import save_file, load_file

from codecarbon import EmissionsTracker

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

        # prepare the model input for instruction tuning
        prompt = "<|AUDIO|> Caption this audio:"
        target_response = text # for example: "This audio contains a humpback"

        # Create the full conversation for training
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_response}
        ]

        # Apply chat template to get the full formatted string
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # Include the assistant response
            enable_thinking=False
        )

        # Tokenize the full conversation
        full_inputs = self.tokenizer(full_text, return_tensors="pt", padding="max_length", max_length=256, truncation=True)

        # Now we need to identify which tokens to compute loss on
        # We only want loss on the assistant's response, not the user prompt

        # Tokenize just the user prompt to find where assistant response starts
        prompt_only = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,  # This adds the assistant prefix
            enable_thinking=False
        )
        prompt_inputs = self.tokenizer(prompt_only, return_tensors="pt", padding="max_length", max_length=256, truncation=True)
        prompt_length = prompt_inputs['input_ids'].shape[1]

        # Create labels: -100 for prompt tokens (ignored), actual token ids for response
        labels = full_inputs['input_ids'].clone()
        labels[0, :prompt_length] = -100  # Ignore loss on prompt tokens

        # Replace <|AUDIO|> token with audio embedding
        audio_token_id = self.tokenizer.convert_tokens_to_ids('<|AUDIO|>')
        audio_token_positions = (full_inputs['input_ids'] == audio_token_id).nonzero(as_tuple=True)

        if len(audio_token_positions[1]) == 0:
            raise ValueError("No <|AUDIO|> token found in the input")

        audio_token_index = audio_token_positions[1][0].item()

        full_inputs['labels'] = labels
        # sqeueeze batch dim
        full_inputs = {k:v.squeeze(0) for k,v in full_inputs.items()}
        # add audio embedding and position
        full_inputs['audio_token_index'] = torch.tensor(audio_token_index).view(-1)
        full_inputs['embedding'] = torch.tensor(embeddings, dtype=torch.float)




        return full_inputs
    

def sample_resolve_nan(x: pd.Series,):
    """
    randomly sample a column that is not nan from the eligible annotation columns
    """

    return_string = ""
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



class AudioCaptionLightningModel(LightningModule):
    def __init__(self, adapter, text_model, tokenizer, learning_rate=1e-4):
        super().__init__()
        self.adapter = adapter
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        # self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, audio_feat, full_inputs, labels):
        # Get embeddings and replace audio token
        audio_token_index = full_inputs.pop("audio_token_index")

        text_embeddings = self.text_model.get_input_embeddings()(full_inputs['input_ids'])
        audio_feat = audio_feat.to(dtype=text_embeddings.dtype, device=text_embeddings.device)
        adapted_audio_embedding = self.adapter(audio_feat)



        # print("Shape1", text_embeddings[:, :audio_token_index, :].shape)
        # print("Shape2", adapted_audio_embedding.unsqueeze(1).shape)
        # print("Shape3", text_embeddings[:, audio_token_index + 1:, :].shape)


        # modified_embeddings = torch.cat([
        #     text_embeddings[:, :audio_token_index, :],
        #     adapted_audio_embedding.unsqueeze(1),
        #     text_embeddings[:, audio_token_index + 1:, :]
        # ], dim=1)

        # to avoid     text_embeddings[:, :audio_token_index, :], TypeError: only integer tensors of a single element can be converted to an index

        batch_size, seq_len, hidden_size = text_embeddings.shape
        # modified_embeddings = torch.zeros((batch_size, seq_len, hidden_size), device=text_embeddings.device, dtype=text_embeddings.dtype)
        # for i in range(batch_size):
        #     idx = audio_token_index[i].item()
        #     modified_embeddings[i, :idx, :] = text_embeddings[i, :idx, :]
        #     modified_embeddings[i, idx, :] = adapted_audio_embedding[i]
        #     if idx + 1 < seq_len:
        #         modified_embeddings[i, idx + 1:, :] = text_embeddings[i, idx + 1:, :]


        # modified_embeddings = text_embeddings.clone()  # This preserves device and dtype
        
        # Replace audio tokens for each item in the batch
        for i in range(batch_size):
            idx = audio_token_index[i].item()
            if 0 <= idx < seq_len:  # Safety check
                text_embeddings[i, idx, :] = adapted_audio_embedding[i]



        # Forward pass with modified embeddings
        model_inputs = {
            'inputs_embeds': text_embeddings,
            'attention_mask': full_inputs['attention_mask'],
            'labels': labels
        }

        outputs = self.text_model(**model_inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        audio_feat = batch.pop("embedding")
        labels = batch.pop("labels")

        outputs = self.forward(audio_feat, batch, labels)
        loss = outputs.loss

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        audio_feat = batch.pop("embedding")
        labels = batch.pop("labels")

        outputs = self.forward(audio_feat, batch, labels)
        loss = outputs.loss

        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.learning_rate)
        return optimizer
    

@hydra.main(version_base=None, config_path="configs", config_name="audio_caption_adapter_config")
def main(cfg: DictConfig):
    """
    Instruction tune and audio adapter
    """


    # model_name = "Qwen/Qwen3-1.7B"
    model_name = cfg.model_name

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cfg.hf_cache_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        cache_dir=cfg.hf_cache_dir,
    )
    model.train()
    model.config.use_cache = False  # important for training

    # extend the tokenizer with special tokens for audio input
    special_tokens_dict = {'additional_special_tokens': ['<|AUDIO|>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    audio_to_text_adapter = AudioToTextAdapter(audio_dim=cfg.audio_dim, text_dim=model.config.hidden_size, bottleneck=cfg.bottleneck_dim).to(model.device)

    ptl_model = AudioCaptionLightningModel(text_model=model, adapter=audio_to_text_adapter, tokenizer=tokenizer, learning_rate=cfg.learning_rate)

    # data
    orcahello_df = pd.read_parquet(cfg.data_path) # contains embeddings_list column and "string_to_embed" column
    
    strings_to_embed = []
    for i, row in orcahello_df.iterrows():
        # print(row)
        out = sample_resolve_nan(row)
        if i< 10:
            print(out)
        strings_to_embed.append(out)

    # get a random audio feature and text
    orcahello_df.loc[:,"string_to_embed"] = strings_to_embed
    # split into train and val:
    orcahello_df.loc[:, 'timestamp'] = pd.to_datetime(orcahello_df['timestamp'], errors='raise', format='%Y-%m-%dT%H:%M:%S.%fZ')

    # print(max_date, min_date)
    print(orcahello_df['timestamp'].min(), orcahello_df['timestamp'].max())


    # split by time
    train_df = orcahello_df[orcahello_df['timestamp'] < pd.Timestamp(cfg.validation_start_date)]
    val_df = orcahello_df[orcahello_df['timestamp'] >= pd.Timestamp(cfg.validation_start_date)]

    print(len(train_df), len(val_df ))
    assert len(train_df)>0, "Train set is empty. Please check the validation_start_date"
    assert len(val_df)>0, "Val set is empty. Please check the validation_start_date"

    training_dataset = OrcaHelloAdapterDataset(train_df, tokenizer)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, shuffle=True, num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    validation_dataset = OrcaHelloAdapterDataset(val_df, tokenizer)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, num_workers=cfg.num_workers, batch_size=cfg.batch_size)


    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best-checkpoint")
    early_stopping = EarlyStopping(monitor="val_loss", patience=cfg.patience, mode="min")


    # training
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        # devices=cfg.devices, # not valid on cpu
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        strategy=cfg.strategy,
        precision=cfg.precision,
        callbacks=[
            model_checkpoint,
            early_stopping,
        ],
    )


    with EmissionsTracker() as tracker:
        trainer.fit(ptl_model, training_dataloader, validation_dataloader)

    # load best model
    best_model_path = trainer.checkpoint_callback.best_model_path

    if os.path.isdir(best_model_path):
        # it is a deepspeed folder, convert it to the regular checkpoint
        from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
        convert_zero_checkpoint_to_fp32_state_dict(best_model_path, os.path.join(best_model_path, "best-checkpoint-final.ckpt"))
        best_model_path = os.path.join(best_model_path, "best-checkpoint-final.ckpt")

    print(f"Best model path: {best_model_path}")

    ptl_model = AudioCaptionLightningModel.load_from_checkpoint(best_model_path, text_model=model, adapter=audio_to_text_adapter, tokenizer=tokenizer, learning_rate=cfg.learning_rate)




    # adapter = ptl_model.adapter



    torch.save(ptl_model.adapter.state_dict(), os.path.join(cfg.output_path, f"adapter-{cfg.text_model_name.replace('/', '-')}.pth"))
    #safetensors
    save_file(ptl_model.adapter.state_dict(), os.path.join(cfg.output_path, f"adapter-{cfg.text_model_name.replace('/', '-')}.safetensors"))





if __name__ == "__main__":
    main()

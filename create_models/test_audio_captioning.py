"""
test_audio_captioning.py


python -u test_audio_captioning.py data_path=/media/bnestor/T7_Black/orcasound_hackathon_2025/orcasound_positive_with_embeddings.parquet devices="auto" num_workers=5 batch_size=8 max_epochs=100 strategy=deepspeed gradient_accumulation_steps=16

"""

from audio_captioning_adapter import AudioToTextAdapter, OrcaHelloAdapterDataset, AudioCaptionLightningModel, sample_resolve_nan

import os
import torch
import pandas as pd
import hydra
from omegaconf import DictConfig

from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file, load_file
import random
import numpy as np

from colorama import Fore, Style




    

@hydra.main(version_base=None, config_path="configs", config_name="audio_caption_adapter_config")
def main(cfg: DictConfig):
    """
    Main function to test the audio captioning model.
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



    audio_to_text_adapter = AudioToTextAdapter(audio_dim=cfg.audio_dim, text_dim=model.config.hidden_size, bottleneck=cfg.bottleneck_dim).float()

    ptl_model = AudioCaptionLightningModel(text_model=model, adapter=audio_to_text_adapter, tokenizer=tokenizer, learning_rate=cfg.learning_rate)

    # data
    orcahello_df = pd.read_parquet(cfg.data_path) # contains embeddings_list column and "string_to_embed" column

    if "description" in orcahello_df.columns:
        orcahello_df = orcahello_df.loc[orcahello_df["description"].notna(), :]
    
    strings_to_embed = []
    for i, row in orcahello_df.iterrows():
        # print(row)
        out = sample_resolve_nan(row)
        if i< 10:
            print(out)
        strings_to_embed.append(out)

    # get a random audio feature and text
    orcahello_df.loc[:,"string_to_embed"] = strings_to_embed
    orcahello_df = orcahello_df.loc[orcahello_df["string_to_embed"].str.strip()!=""]
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
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, num_workers=cfg.num_workers, batch_size=1)


    # best_model_path = "lightning_logs/version_23/checkpoints/best-checkpoint.ckpt"
    best_model_path = "lightning_logs/version_47/checkpoints/best-checkpoint.ckpt/best-checkpoint-final.ckpt"


    if os.path.isdir(best_model_path):
        # it is a deepspeed folder, convert it to the regular checkpoint
        from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
        convert_zero_checkpoint_to_fp32_state_dict(best_model_path, os.path.join(best_model_path, "best-checkpoint-final.ckpt"))
        best_model_path = os.path.join(best_model_path, "best-checkpoint-final.ckpt")

    print(f"Best model path: {best_model_path}")

    try:
        ptl_model = AudioCaptionLightningModel.load_from_checkpoint(best_model_path, text_model=model, adapter=audio_to_text_adapter, tokenizer=tokenizer, learning_rate=cfg.learning_rate)
    except Exception:
        # getting error missing keys "text_model.model.embed_tokens.weight"
        print(sorted(ptl_model.state_dict().keys())[:15])
        state_dict = torch.load(best_model_path, map_location='cpu')['state_dict']
        # state_dict = load_file(best_model_path)
        print(sorted(state_dict.keys())[:15])
        ptl_model.load_state_dict(state_dict, strict=False)

        





    # adapter = ptl_model.adapter



    torch.save(ptl_model.adapter.state_dict(), os.path.join(cfg.output_path, f"adapter-{cfg.model_name.replace('/', '-')}.pth"))
    #safetensors
    save_file(ptl_model.adapter.state_dict(), os.path.join(cfg.output_path, f"adapter-{cfg.model_name.replace('/', '-')}.safetensors"))



    # now lets go through and do some generations, comparing to the original labels:

    # cast all ptl_model params to float32
    ptl_model = ptl_model.float()

    ptl_model.eval()
    with torch.no_grad():
        for batch in validation_dataloader:
            audio_feat = batch.pop("embedding")
            labels = batch.pop("labels")

            # decode the batch["input_ids"][0] to see the input
            # print(Fore.BLUE +"input ids" + Style.RESET_ALL, tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False))

            outputs, model_inputs = ptl_model(audio_feat, batch, labels, snap_to_nearest_token=False, return_model_inputs=True)

            logits = outputs.logits
            # get the predicted tokens
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            # compare generated_text with text_inputs
            # to avoid out of range error, we will replace -100 in labels with tokenizer.pad_token_id
            labels[labels == -100] = tokenizer.pad_token_id

            print("---"*50)
            print(Fore.CYAN + "Original: " +Style.RESET_ALL, tokenizer.decode(labels[0], skip_special_tokens=True))
            # decode model_inputs["inputs_embeds"][0] to see the input
            #input_embeds to ids:
            input_ids = torch.argmax(model_inputs["inputs_embeds"][0] @ ptl_model.text_model.get_input_embeddings().weight.T, dim=-1)
            print(Fore.YELLOW + "Input: " + Style.RESET_ALL, tokenizer.decode(input_ids, skip_special_tokens=True).strip().replace("\n"," ").replace("  "," "))
            print(Fore.MAGENTA + "Generated: " + Style.RESET_ALL, tokenizer.decode(predicted_tokens[0], skip_special_tokens=True).strip().replace("\n"," ").replace("  "," "))


def short_test():
    print(Fore.GREEN + "Running short test..." + Style.RESET_ALL)
    # Implement short test logic here


if __name__ == "__main__":
    main()

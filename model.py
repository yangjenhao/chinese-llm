# -*- coding: utf-8 -*-
"""
@ creater : JenHao
"""
# -------------------------------------------------------------------------
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
# -------------------------------------------------------------------------
import os
import sys
import argparse
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (PeftModel, prepare_model_for_int8_training, LoraConfig, get_peft_model, get_peft_model_state_dict,)
from config import parse_config


# -------------------------------------------------------------------------
#? ***** lora settings *****
MICRO_BATCH_SIZE = 4               # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 6                          # we don't always need 3 tbh
LEARNING_RATE = 3e-4                # the Karpathy constant
CUTOFF_LEN = 1024                    # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 0
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
# -------------------------------------------------------------------------
def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }    

def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""下方是一個關於任務的指令，以及一個提供與任務相關之資訊的輸入。請撰寫一個能適當地完成該任務需求的回覆。
### 指令:
{data_point["instruction"]}
### 輸入:
{data_point["input"]}
### 回覆:
"""

        )
        if data_point["input"]
        else (
            f"""下方是一個關於任務的指令。請撰寫一個能適當地完成該任務需求的回覆。
### 輸入:
{data_point["instruction"]}
### 回覆:
"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        )
        - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }

if __name__ == '__main__':
    print('\n ***** __main__ start here ***** ')
    args = parse_config()
    
    #? ***** 儲存參數檔案 *****
    if not os.path.exists(args.output_dir):  #判斷存放 "model/當前的model"的資料夾是否存在
        os.makedirs(args.output_dir)
    log_path = os.path.join(os.path.expanduser(args.log_dir), 'argument.txt')
    with open(log_path, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n'%(key, value))
    
    #? ***** main logic starts here *****
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        device_map=device_map,
        cache_dir=args.cache_dir
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name,
        add_eos_token=True,
        cache_dir=args.cache_dir
    )

    if args.from_ckpt:
        # Resume from checkpoint
        model = PeftModel.from_pretrained(model, args.ckpt_name)

    model = prepare_model_for_int8_training(model)
    
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset(
        "json",
        data_files=args.dataset_dir,
        cache_dir=args.cache_dir
    )

    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=args.num_epoch,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=args.logging_steps,
            # evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            save_strategy="steps",
            # eval_steps=200 if VAL_SET_SIZE > 0 else None,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit,
            # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=args.report_to,
            run_name=args.wandb_run_name if args.report_to == 'wandb' else None,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != 'win32':
        model = torch.compile(model)
        
    #? ***** main train here *****
    trainer.train()
    model.save_pretrained(args.output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")
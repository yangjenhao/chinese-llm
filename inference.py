# -*- coding: utf-8 -*-
"""
@ creater : JenHao
"""

# -------------------------------------------------------------------------
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -------------------------------------------------------------------------
import torch
import json
import time
import re
import unicodedata
import argparse
from colorama import Fore, Style
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, set_seed

def to_jsonfile(indexs, links, questions, answers , eval_texts, save_path):    
    """
    存成指定的json格式
    """
    data = []
    for index, link, question, answer ,eval_text in zip(indexs, links, questions, answers, eval_texts):
        item = {"id" : index, "link": link, "input": question, "output": answer, "llm_text" : eval_text}
        # item = {"instruction": '你現在是一個摘要機器人，請幫我摘要以下這篇文章。', "input": question, "output": answer}
        data.append(item)
    
    if os.path.exists(save_path):
        with open(save_path, 'r') as file:
            existing_data = json.load(file)    
        merged_data = existing_data + data
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
    else :
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def remove_question_marks(input_string):
    cleaned_string = input_string.replace("�", "")  # 将问号方块替换为空字符串
    return cleaned_string

def remove_after_fullwidth_english(input_string):
    pattern = r'[Ａ-Ｚａ-ｚ]+'
    matches = re.finditer(pattern, input_string)
    positions = [match.start() for match in matches]
    if positions:
        min_position = min(positions)
        new_string = input_string[:min_position]
    else:
        new_string = input_string

    return new_string
      
def set_seeds(seed):
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_prompt(instruction, input=None):
    # sorry about the formatting disaster gotta move fast
    if input:
        return f"""下方是一個關於任務的指令，以及一個提供與任務相關之資訊的輸入。請撰寫一個能適當地完成該任務指令需求的回覆。
### 指令:
{instruction}

### 輸入:
{input}

### 回覆:"""
    else:
        return f"""下方是一個關於任務的指令。請撰寫一個能適當地完成該任務指令需求的回覆。
### 輸入:
{instruction}

### 回覆:"""

# def evaluate(instruction, generation_config, input=None):
# def evaluate(instruction, generation_config, input='你現在是一個摘要機器人，請幫我摘要以下這篇文章'):
def evaluate(instruction, generation_config, input='請根據下方標題，以你在醫療領域的知識和想像力，生成一篇與醫療相關的文章。'):
    # prompt = generate_prompt(instruction, input)
    
    #####? have instruction/input 
    prompt = generate_prompt(input, instruction)
    print()
    print(prompt)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        # print(f"{Fore.GREEN}回覆:{Style.RESET_ALL}")
        # print(output.split("### 回覆:")[1].strip() + '\n')
    print(output.split("### 回覆:")[1].strip())
    return output.split("### 回覆:")[1].strip()


def parse_config():
    parser = argparse.ArgumentParser()
    #####? < 要更改的地方 > #####
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, required=True)
    parser.add_argument("--currrent", type=int, default=0)
    parser.add_argument("--eval_data", type=str, required=True)
    #####? </> ##### 
    
    parser.add_argument("--cache_dir", type=str, default="../cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.65)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_config()
    # set_seeds(args.seed)
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir
    )
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        # device_map="auto",
        device_map={'': 0},
        cache_dir=args.cache_dir
    )
    # load from checkpoint
    model = PeftModel.from_pretrained(model, args.ckpt_name, device_map={'': 0})

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
   
    # while(True):
    #     x = evaluate(input(f"\n{'-'*10}\n{Fore.BLUE}指令: {Style.RESET_ALL}"), generation_config)
    #     print(x)
        

    # with open('/workplace/jhyang/NCULLM/Data/EVAL/eval.json', 'r', encoding='utf-8') as file:
    #     data = json.load(file)
    with open(args.eval_data, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    eval_id, eval_link, eval_input, eval_output = [], [], [], []
    for article in data:
            eval_id.append(article['id'])
            eval_link.append(article['link'])
            eval_input.append(article['input'])
            eval_output.append(article['output'])
        
    eval_text = []
    json_id, json_link, json_input, json_output = [], [], [], []
    ii = args.currrent
    while(True):
        # text = evaluate(input(f"\n{'-'*10}\n{Fore.BLUE}指令: {Style.RESET_ALL}"), generation_config)
        ###? llm output
        text = evaluate(eval_input[ii] , generation_config)
        parts = text.split("</s>")
        new_string = parts[0]
        new_string = remove_question_marks(new_string)
        new_string = remove_after_fullwidth_english(new_string)
        
        ###? path name
        path = args.ckpt_name
        last_slash_index = path.rfind("/")  
        path_name = path[last_slash_index + 1:]
        
        eval_text.append(new_string)
        json_id.append(eval_id[ii])
        json_link.append(eval_link[ii])
        json_input.append(eval_input[ii])
        json_output.append(eval_output[ii])
        to_jsonfile(json_id, json_link, json_input, json_output, eval_text,'{}.json'.format(path_name))
        print('current i = ', ii)
        break
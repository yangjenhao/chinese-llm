# LLM使用手冊

## 檔案介紹區：
1. `data/` : 訓練資料存放區，輸入檔案格式可以參考`ref/traditional-chinese-alpaca/data/alpaca-tw_en_instruction.json`
2. `model/` : 存放模型的地方
3. `ref/` : 原始參考github來源
4. `config.py` : model.py的參數放置區
5. `inference.py` : 模型訓練完後，使用推理區
6. `model.py` : 模型訓練py檔
 
## 可使用的上游model，可自行挖掘更好用的:
huggingface : shibing624/chinese-alpaca-plus-13b-hf (效果還行)
huggingface : minlik/chinese-alpaca-13b-merged (效果普普)

## 環境參考(218上):

A100-PCIE-40GB / CUDA Version: 11.1
```
pip install peft==0.2.0
pip install sentencepiece
pip install accelerate==0.20.3
pip install bitsandbytes==0.37.1
pip install datasets
pip install wandb
pip install transformers==4.27.2
pip install evaluate
```

## 使用方法：
1. 訓練模型(更改參數後):

```
bash run_model.sh
```

2. 推論(更改參數後):
```
bash run_inference.sh
```
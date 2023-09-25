
for ((j=0; j<=5; j++))
do
    for ((i=0; i<=2; i++))
    do
        python inference.py \
        --model_name='shibing624/chinese-alpaca-plus-13b-hf' \
        --ckpt_name='/workplace/jhyang/NCULLM/G-model/G-articles-1024' \
        --eval_data='/workplace/jhyang/NCULLM/Data/EVAL/article/eval_article.json' \
        --currrent=${j}
    done
    # --ckpt_name='/workplace/jhyang/NCULLM/NCULLM_workplace3/model/chinese-alpaca-plus-13b-h-G-articles-1024' \
    # --eval_data='/workplace/jhyang/NCULLM/Data/EVAL/article/eval_G-summary-1024_MyTitle.json' \
done
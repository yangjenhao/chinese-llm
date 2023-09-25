# -*- coding: utf-8 -*-
"""
@ creater : JenHao
"""

import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--from_ckpt", action="store_true")
    parser.add_argument("--ckpt_name", type=str, required=True)
    

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="../cache")
    parser.add_argument('--log_dir', type=str, required=True)


    parser.add_argument("--num_epoch", type=int, required=True)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=600)
    parser.add_argument("--save_total_limit", type=int, default=3)


    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_run_name", type=str, required=True)

    return parser.parse_args()


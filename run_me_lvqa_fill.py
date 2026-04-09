#!/usr/bin/env python3
"""Entry point for the train-only ME-LVQA filler."""

import argparse
import os

from me_lvqa_util.lvqa_inference import fill_target_file
from me_lvqa_util.lvqa_reference import build_reference_profiles
from me_lvqa_util.lvqa_priors import build_dataset_statistics
from me_lvqa_util.lvqa_requirements import parse_question_type_target
from model.qwen_bridge import QwenBridge
from util.io_utils import load_jsonl


def main():
    parser = argparse.ArgumentParser(description="Phase-0 train-only ME-LVQA baseline filler.")
    parser.add_argument("--train", default=os.path.join("Q&A_jsonl", "me_lvqa_samm_casme3.jsonl"), help="Training JSONL (answered rows).")
    parser.add_argument("--casme3-test", default=os.path.join("Q&A_jsonl", "me_lvqa_casme3_test_to_answer.jsonl"), help="CASME3 target JSONL.")
    parser.add_argument("--samm-test", default=os.path.join("Q&A_jsonl", "me_lvqa_samm_test_to_answer.jsonl"), help="SAMM target JSONL.")
    parser.add_argument("--outdir", default="generated_answers_train_only", help="Output directory.")
    parser.add_argument("--seed", type=int, default=202601, help="Deterministic seed.")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-7B-Instruct", help="Required Qwen model name or local path.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128, help="Max new tokens per Qwen call.")
    args = parser.parse_args()

    qwen_bridge = QwenBridge(
        model_name=args.qwen_model,
        max_new_tokens=args.qwen_max_new_tokens,
    )

    train_rows = load_jsonl(args.train)
    reference_profiles = build_reference_profiles(train_rows)
    stats_by_dataset = build_dataset_statistics(reference_profiles, train_rows)
    target_question_parser = lambda q: parse_question_type_target(q, qwen_bridge=qwen_bridge)

    casme3_out = os.path.join(args.outdir, "me_lvqa_casme3_test_pred.jsonl")
    samm_out = os.path.join(args.outdir, "me_lvqa_samm_test_pred.jsonl")

    fill_target_file(
        target_path=args.casme3_test,
        output_path=casme3_out,
        dataset_hint="casme3",
        reference_profiles=reference_profiles,
        stats_by_dataset=stats_by_dataset,
        seed=args.seed,
        question_parser=target_question_parser,
    )
    fill_target_file(
        target_path=args.samm_test,
        output_path=samm_out,
        dataset_hint="samm",
        reference_profiles=reference_profiles,
        stats_by_dataset=stats_by_dataset,
        seed=args.seed,
        question_parser=target_question_parser,
    )


if __name__ == "__main__":
    main()

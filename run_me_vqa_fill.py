#!/usr/bin/env python3
"""Thin entry point for ME-VQA train-only heuristic filler."""

import argparse
import os

from me_vqa_util.vqa_inference import fill_target_file
from util.io_utils import load_jsonl
from me_vqa_util.vqa_priors import build_priors
from me_vqa_util.vqa_reference import build_reference_profiles
from me_vqa_util.vqa_requirements import parse_question_target
from model.qwen_bridge import QwenBridge


def main():
    parser = argparse.ArgumentParser(description="Train-only heuristic filler for ME-VQA v2 schema.")
    parser.add_argument("--reference", default=os.path.join("Q&A_jsonl", "me_vqa_samm_casme2_smic_v2.jsonl"), help="Reference JSONL with labeled reference rows.")
    parser.add_argument("--casme3-test", default=os.path.join("Q&A_jsonl", "me_vqa_casme3_v2_test_to_answer.jsonl"), help="CASME3 test JSONL to fill.")
    parser.add_argument("--samm-test", default=os.path.join("Q&A_jsonl", "me_vqa_samm_v2_test_to_answer.jsonl"), help="SAMM test JSONL to fill.")
    parser.add_argument("--outdir", default="generated_answers_me_vqa_train_only", help="Output directory.")
    parser.add_argument("--seed", type=int, default=202601, help="Deterministic seed.")
    parser.add_argument("--use-qwen", action="store_true", help="Use optional Qwen assist for target parsing and analysis rendering.")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-7B-Instruct", help="Qwen model name or local path.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128, help="Max new tokens per Qwen call.")
    args = parser.parse_args()

    qwen_bridge = None
    if args.use_qwen:
        qwen_bridge = QwenBridge(
            model_name=args.qwen_model,
            max_new_tokens=args.qwen_max_new_tokens,
        )

    reference_rows = load_jsonl(args.reference)
    reference_profiles = build_reference_profiles(reference_rows)
    stats_by_dataset = build_priors(reference_profiles)
    target_question_parser = lambda q: parse_question_target(q, qwen_bridge=qwen_bridge)
    analysis_renderer = None
    if qwen_bridge is not None:
        analysis_renderer = qwen_bridge.render_mevqa_analysis

    casme3_out = os.path.join(args.outdir, "me_vqa_casme3_v2_test_pred.jsonl")
    samm_out = os.path.join(args.outdir, "me_vqa_samm_v2_test_pred.jsonl")

    fill_target_file(
        target_path=args.casme3_test,
        output_path=casme3_out,
        dataset_hint="casme3",
        reference_profiles=reference_profiles,
        stats_by_dataset=stats_by_dataset,
        seed=args.seed,
        question_parser=target_question_parser,
        analysis_renderer=analysis_renderer,
    )
    fill_target_file(
        target_path=args.samm_test,
        output_path=samm_out,
        dataset_hint="samm",
        reference_profiles=reference_profiles,
        stats_by_dataset=stats_by_dataset,
        seed=args.seed,
        question_parser=target_question_parser,
        analysis_renderer=analysis_renderer,
    )


if __name__ == "__main__":
    main()

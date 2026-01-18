#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ours Model Evaluation Script + Bootstrap Statistical Significance Analysis

"""

import os
import yaml
import torch
import editdistance
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
from datetime import datetime
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statistical_significance import StatisticalSignificance


from models.recognizer import EnhancedIndustrialOCR
from utils.dataset import MeterDataset
from utils.collate import CollateFN

# ========== Bootstrap ==========
N_BOOTSTRAP = 30  # Bootstrap


def normalize_digits_only(s: str) -> str:
    """Keep only digit characters in string"""
    return "".join(ch for ch in s if ch.isdigit())


def evaluate_model_bootstrap(
        model,
        test_loader,
        idx2char,
        device,
        max_len,
        sos_idx,
        eos_idx,
        pad_idx,
        bootstrap_id,
        coco_dataset=None
):
    """
    Bootstrap evaluate model

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_ground_truths = []
    all_image_ids = []
    all_filenames = []

    print(f"\n🚀 Bootstrap {bootstrap_id} inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Bootstrap {bootstrap_id}")):
            if not batch:
                continue

            # Use correct batch key names
            src = batch["image"].to(device)
            src_img_mask = batch["mask"].to(device)
            ground_truth_labels = batch["label"]
            image_ids = batch["image_id"]

            try:
                # Use model's predict method for inference
                predicted_indices = model.predict(
                    src, src_img_mask, max_len, sos_idx, eos_idx, pad_idx
                )

                # Decode prediction results
                for idx_sequence, gt_label, img_id in zip(predicted_indices, ground_truth_labels, image_ids):
                    pred_chars = []
                    for idx_tensor in idx_sequence:
                        idx = idx_tensor.item()
                        if idx == eos_idx:
                            break
                        if idx == pad_idx or idx == sos_idx:
                            continue
                        pred_chars.append(idx2char.get(idx, '?'))

                    pred_text = "".join(pred_chars)
                    all_predictions.append(pred_text)
                    all_ground_truths.append(gt_label)
                    all_image_ids.append(img_id)

                    # Get image filename
                    if coco_dataset and img_id in coco_dataset.imgs:
                        filename = coco_dataset.imgs[img_id]['file_name']
                    else:
                        filename = f"image_{img_id}"
                    all_filenames.append(filename)

            except Exception as e:
                print(f"\nWarning: batch {batch_idx} inference failed: {e}")
                # Add empty predictions for this batch
                for gt_label, img_id in zip(ground_truth_labels, image_ids):
                    all_predictions.append("")
                    all_ground_truths.append(gt_label)
                    all_image_ids.append(img_id)

                    if coco_dataset and img_id in coco_dataset.imgs:
                        filename = coco_dataset.imgs[img_id]['file_name']
                    else:
                        filename = f"image_{img_id}"
                    all_filenames.append(filename)
                continue

    # Calculate metrics
    exact_raw = 0
    exact_digits = 0
    cer_raw_sum = 0.0
    cer_digits_sum = 0.0
    total_count = len(all_predictions)

    detailed_results = []

    for filename, img_id, pred, gt in zip(all_filenames, all_image_ids, all_predictions, all_ground_truths):
        # Normalize to digits only
        pred_digits = normalize_digits_only(pred)
        gt_digits = normalize_digits_only(gt)

        # Calculate ExactMatch
        is_exact_raw = (pred == gt)
        is_exact_digits = (pred_digits == gt_digits)

        exact_raw += int(is_exact_raw)
        exact_digits += int(is_exact_digits)

        # Calculate CER
        edit_dist_raw = editdistance.eval(pred, gt)
        edit_dist_digits = editdistance.eval(pred_digits, gt_digits)

        cer_raw = edit_dist_raw / len(gt) if len(gt) > 0 else (1.0 if len(pred) > 0 else 0.0)
        cer_digits = edit_dist_digits / len(gt_digits) if len(gt_digits) > 0 else (1.0 if len(pred_digits) > 0 else 0.0)

        cer_raw_sum += cer_raw
        cer_digits_sum += cer_digits

        # Save detailed results
        detailed_results.append({
            'image_name': filename,
            'image_id': img_id,
            'gt': gt,
            'pred': pred,
            'gt_digits': gt_digits,
            'pred_digits': pred_digits,
            'exact_match_raw': is_exact_raw,
            'exact_match_digits': is_exact_digits,
            'cer_raw': round(cer_raw, 4),
            'cer_digits': round(cer_digits, 4)
        })

    # Calculate average metrics
    exact_match_raw = exact_raw / total_count if total_count > 0 else 0
    exact_match_digits = exact_digits / total_count if total_count > 0 else 0
    avg_cer_raw = cer_raw_sum / total_count if total_count > 0 else 0
    avg_cer_digits = cer_digits_sum / total_count if total_count > 0 else 0

    print(f"[Bootstrap {bootstrap_id}] ExactMatch(raw)={exact_match_raw:.4f}, "
          f"ExactMatch(digits)={exact_match_digits:.4f}, "
          f"CER(raw)={avg_cer_raw:.4f}, CER(digits)={avg_cer_digits:.4f}")

    return {
        'bootstrap_run': bootstrap_id,
        'total_samples': total_count,
        'exact_match_raw': exact_match_raw,
        'exact_match_digits': exact_match_digits,
        'avg_cer_raw': avg_cer_raw,
        'avg_cer_digits': avg_cer_digits,
        'exact_match_raw_count': exact_raw,
        'exact_match_digits_count': exact_digits,
        'detailed_results': detailed_results
    }


def bootstrap_sample_annotations(original_ann_path: str, seed: int):
    """
    Bootstrap resample COCO annotations

    Returns:
        dict: COCO data after Bootstrap sampling
    """
    # Load original annotations
    with open(original_ann_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    n_samples = len(images)

    # Bootstrap resampling
    np.random.seed(seed)
    bootstrap_indices = np.random.choice(
        n_samples,
        size=n_samples,
        replace=True  # sampling with replacement
    )

    # Create Bootstrap data
    bootstrap_images = []
    bootstrap_annotations = []

    for new_id, old_idx in enumerate(bootstrap_indices):
        old_image = images[old_idx]
        old_image_id = old_image['id']

        # New image record
        new_image = old_image.copy()
        new_image['id'] = new_id
        bootstrap_images.append(new_image)

        # Find corresponding annotation
        for ann in annotations:
            if ann['image_id'] == old_image_id:
                new_ann = ann.copy()
                new_ann['id'] = len(bootstrap_annotations)
                new_ann['image_id'] = new_id
                bootstrap_annotations.append(new_ann)
                break

    return {
        'images': bootstrap_images,
        'annotations': bootstrap_annotations,
        'categories': coco_data.get('categories', [])
    }


def main():
    """- Bootstrap resamplingEvaluation"""

    parser = argparse.ArgumentParser(description='Bootstrap resamplingEvaluation')
    parser.add_argument('--config', default='./xx/configs', help='Configuration file path')
    parser.add_argument('--checkpoint',
                        default='path/to/xx.pth',
                        help='model checkpoint path')
    parser.add_argument('--ann_path',
                        default='./outputs/public_data/split/train/annotations.json',
                        help='Specify the file path')
    parser.add_argument('--img_dir',
                        default='./outputs/public_data/split/train/images',
                        help='img_dir')
    parser.add_argument('--output_dir', default='./bootstrap_results', help='output_dir')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== Load config ==========
    print("=" * 80)
    print("Bootstrap resamplingEvaluation")
    print("=" * 80)
    print(f"📋 Load config...")

    config_dir = Path(args.config)
    model_config_path = config_dir / "model.yaml"
    data_config_path = config_dir / "data.yaml"

    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)

    with open(data_config_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    # ========== Build vocabulary ==========
    print("📖 Build vocabulary...")
    chars = data_cfg.get("characters", "")
    special_tokens = data_cfg.get("special_tokens", [])
    max_len_attention = model_cfg.get("max_len", 19)
    is_grayscale = (model_cfg.get("input_channels", 3) == 1)

    attention_special_tokens = [st for st in special_tokens if st != "<BLK>"]
    attention_vocab_list = list(chars) + attention_special_tokens
    char2idx_attention = {char: i for i, char in enumerate(attention_vocab_list)}
    idx2char_attention = {i: char for char, i in char2idx_attention.items()}
    vocab_size_attention = len(char2idx_attention)

    pad_idx = char2idx_attention.get("<PAD>", len(char2idx_attention) - 1)
    sos_idx = char2idx_attention.get("<SOS>", 0)
    eos_idx = char2idx_attention.get("<EOS>", 1)

    char2idx_ctc = {char: i for i, char in enumerate(list(chars) + special_tokens)}
    vocab_size_ctc = len(char2idx_ctc)
    blank_idx_ctc = char2idx_ctc.get("<BLK>")

    model_cfg["pad_idx"] = pad_idx

    print(f"   Vocabulary size: Attention={vocab_size_attention}, CTC={vocab_size_ctc}")


    print("\n🔨 Initialize model...")
    model = EnhancedIndustrialOCR(
        vocab_size=vocab_size_attention,
        vocab_size_ctc=vocab_size_ctc,
        model_cfg=model_cfg,
        blank_idx_ctc=blank_idx_ctc
    ).to(device)


    print(f"\n📥 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle DataParallel
    if any(key.startswith('module.') for key in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    print("✅ Checkpoint loaded successfully")

    # ========== Bootstrap resamplingEvaluation ==========
    print(f"\n📊 Starting Bootstrap evaluation (n={N_BOOTSTRAP})")

    exact_match_raw_list = []
    exact_match_digits_list = []
    cer_raw_list = []
    cer_digits_list = []
    all_runs = {}

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N_BOOTSTRAP):
        print(f"\n{'=' * 80}")
        print(f"Bootstrap run {i + 1}/{N_BOOTSTRAP}")
        print('=' * 80)


        bootstrap_coco = bootstrap_sample_annotations(args.ann_path, seed=42 + i)

        # Temporarily save Bootstrap annotations
        temp_ann_path = output_dir / f"temp_bootstrap_{i + 1}_annotations.json"
        with open(temp_ann_path, 'w', encoding='utf-8') as f:
            json.dump(bootstrap_coco, f, ensure_ascii=False, indent=2)


        try:
            test_set = MeterDataset(
                img_dir=args.img_dir,
                ann_path=str(temp_ann_path),
                chars=chars,
                max_len=max_len_attention,
                special_tokens=special_tokens,
                transform=None,
                is_grayscale=is_grayscale,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                grouping_config=None
            )

            collate_fn = CollateFN(
                batch_size=args.batch_size,
                char2idx_ctc=char2idx_ctc,
                max_collate_h=model_cfg.get("max_collate_h", 650),
                max_collate_w=model_cfg.get("max_collate_w", 800),
                target_feature_map_size=(model_cfg.get("feature_map_h"), model_cfg.get("feature_map_w"))
            )

            test_loader = DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )

            print(f"   Bootstrap {i + 1} sample count: {len(test_set)}")

        except Exception as e:
            print(f"❌ Bootstrap {i + 1} dataset creation failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Evaluation
        metrics = evaluate_model_bootstrap(
            model=model,
            test_loader=test_loader,
            idx2char=idx2char_attention,
            device=device,
            max_len=max_len_attention,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            bootstrap_id=i + 1,
            coco_dataset=test_set.coco
        )

        exact_match_raw_list.append(metrics['exact_match_raw'])
        exact_match_digits_list.append(metrics['exact_match_digits'])
        cer_raw_list.append(metrics['avg_cer_raw'])
        cer_digits_list.append(metrics['avg_cer_digits'])

        all_runs[f"bootstrap_{i + 1}"] = metrics


        temp_ann_path.unlink()


    print("\n" + "=" * 80)
    print("Statistical Significance Analysis")
    print("=" * 80)

    stat_sig = StatisticalSignificance()

    # Calculate mean and standard deviation
    exact_match_raw_mean = float(np.mean(exact_match_raw_list))
    exact_match_raw_std = float(np.std(exact_match_raw_list, ddof=1)) if N_BOOTSTRAP > 1 else 0.0
    exact_match_digits_mean = float(np.mean(exact_match_digits_list))
    exact_match_digits_std = float(np.std(exact_match_digits_list, ddof=1)) if N_BOOTSTRAP > 1 else 0.0
    cer_raw_mean = float(np.mean(cer_raw_list))
    cer_raw_std = float(np.std(cer_raw_list, ddof=1)) if N_BOOTSTRAP > 1 else 0.0
    cer_digits_mean = float(np.mean(cer_digits_list))
    cer_digits_std = float(np.std(cer_digits_list, ddof=1)) if N_BOOTSTRAP > 1 else 0.0

    # Bootstrap置信区间
    print("\nBootstrap Confidence Interval Analysis:")
    print("-" * 80)

    bootstrap_em_raw = stat_sig.bootstrap_confidence_interval(exact_match_raw_list, n_bootstrap=10000)
    print(f"ExactMatch(raw):")
    print(f"  Mean: {bootstrap_em_raw['mean']:.4f}")
    print(
        f"  95% CI: [{bootstrap_em_raw['confidence_interval'][0]:.4f}, {bootstrap_em_raw['confidence_interval'][1]:.4f}]")

    bootstrap_em_digits = stat_sig.bootstrap_confidence_interval(exact_match_digits_list, n_bootstrap=10000)
    print(f"\nExactMatch(digits):")
    print(f"  Mean: {bootstrap_em_digits['mean']:.4f}")
    print(
        f"  95% CI: [{bootstrap_em_digits['confidence_interval'][0]:.4f}, {bootstrap_em_digits['confidence_interval'][1]:.4f}]")

    bootstrap_cer_raw = stat_sig.bootstrap_confidence_interval(cer_raw_list, n_bootstrap=10000)
    print(f"\nCER(raw):")
    print(f"  Mean: {bootstrap_cer_raw['mean']:.4f}")
    print(
        f"  95% CI: [{bootstrap_cer_raw['confidence_interval'][0]:.4f}, {bootstrap_cer_raw['confidence_interval'][1]:.4f}]")

    bootstrap_cer_digits = stat_sig.bootstrap_confidence_interval(cer_digits_list, n_bootstrap=10000)
    print(f"\nCER(digits):")
    print(f"  Mean: {bootstrap_cer_digits['mean']:.4f}")
    print(
        f"  95% CI: [{bootstrap_cer_digits['confidence_interval'][0]:.4f}, {bootstrap_cer_digits['confidence_interval'][1]:.4f}]")

    # ========== Print final results ==========
    print("\n" + "=" * 80)
    print("Final Results (mean ± std)")
    print("=" * 80)
    print(f"ExactMatch(raw)   : {exact_match_raw_mean:.4f} ± {exact_match_raw_std:.4f}")
    print(f"ExactMatch(digit) : {exact_match_digits_mean:.4f} ± {exact_match_digits_std:.4f}")
    print(f"Mean CER(raw)     : {cer_raw_mean:.4f} ± {cer_raw_std:.4f}")
    print(f"Mean CER(digit)   : {cer_digits_mean:.4f} ± {cer_digits_std:.4f}")
    print("=" * 80)

    # ========== Save results ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ours_results_with_stats_{timestamp}.json"

    results_data = {
        "n_bootstrap": N_BOOTSTRAP,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "test_dataset": {
            "annotation_file": args.ann_path,
            "image_dir": args.img_dir
        },
        "statistical_summary": {
            "exact_match_raw": {
                "mean": exact_match_raw_mean,
                "std": exact_match_raw_std,
                "values": [float(v) for v in exact_match_raw_list],
                "bootstrap_ci_95": bootstrap_em_raw['confidence_interval']
            },
            "exact_match_digits": {
                "mean": exact_match_digits_mean,
                "std": exact_match_digits_std,
                "values": [float(v) for v in exact_match_digits_list],
                "bootstrap_ci_95": bootstrap_em_digits['confidence_interval']
            },
            "cer_raw": {
                "mean": cer_raw_mean,
                "std": cer_raw_std,
                "values": [float(v) for v in cer_raw_list],
                "bootstrap_ci_95": bootstrap_cer_raw['confidence_interval']
            },
            "cer_digits": {
                "mean": cer_digits_mean,
                "std": cer_digits_std,
                "values": [float(v) for v in cer_digits_list],
                "bootstrap_ci_95": bootstrap_cer_digits['confidence_interval']
            }
        },
        "runs": all_runs,
        "timestamp": timestamp
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print("\nEvaluation完成！")


if __name__ == '__main__':
    main()
# OCR/scripts/train.py
import random
import torch
import yaml
import os
import time
import editdistance
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torch.cuda.amp import GradScaler, autocast
from utils.loss_functions import FocalLoss
from models.recognizer import EnhancedIndustrialOCR
from utils.augmentation import MeterAugment
from utils.collate import CollateFN
from utils.dataset import MeterDataset
from torch.nn import CrossEntropyLoss, CTCLoss, BCEWithLogitsLoss
from experiment_logger import initialize_log_file, log_epoch_data
import torchvision.utils as vutils



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

def main():
    global best_epoch
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        config_base_path = "./configs/"  # Please ensure this is your config file base path
        with open(os.path.join(config_base_path, "data.yaml"), encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
        with open(os.path.join(config_base_path, "model.yaml"), encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
        print("Config files loaded。")
    except Exception as e:
        print(f"Error loading config files: {e}")
        return

    chars = data_cfg.get("characters", "")
    special_tokens = data_cfg.get("special_tokens", [])
    max_len_attention = model_cfg.get("max_len", 19)

    d_model = model_cfg.get('d_model', 256)
    batch_size = model_cfg.get('batch_size', 4)
    num_workers = model_cfg.get("num_workers", 2)

    pce_vis = model_cfg.get("pce_vis", False)
    epoch_to_visualize_val = model_cfg.get("epoch_vis", 0)

    model_type = model_cfg.get("model_type", "attention_only")
    lambda_ctc = float(model_cfg.get("lambda_ctc", 0.2)) if model_type == "hybrid" else 0.0

    use_segmentation_module = model_cfg.get('use_segmentation_module', False)
    lambda_seg = float(model_cfg.get('lambda_seg', 0.2)) if use_segmentation_module else 0.0

    is_grayscale_dataset = (model_cfg.get("input_channels", 3) == 1)
    print(f"Images will be loaded as {'grayscale' if is_grayscale_dataset else 'color'} loaded。")


    u_penalty_target_min = model_cfg.get('u_penalty_target_min', 0.05)
    u_penalty_target_max = model_cfg.get('u_penalty_target_max', 0.95)
    u_penalty_weight = model_cfg.get('u_penalty_weight', 0.1)

    focal_loss_alpha = model_cfg.get('focal_loss_alpha', 0.25)
    focal_loss_gamma = model_cfg.get('focal_loss_gamma', 2.0)


    attention_special_tokens = [st for st in special_tokens if st != "<BLK>"]
    attention_vocab_list = list(chars) + attention_special_tokens
    char2idx_attention = {char: i for i, char in enumerate(attention_vocab_list)}
    idx2char_attention = {i: char for char, i in char2idx_attention.items()}
    vocab_size_attention = len(char2idx_attention)
    pad_idx_attention = char2idx_attention["<PAD>"]
    sos_idx_attention = char2idx_attention["<SOS>"]
    eos_idx_attention = char2idx_attention["<EOS>"]

    char2idx_ctc = {char: i for i, char in enumerate(list(chars) + special_tokens)}

    vocab_size_ctc = len(char2idx_ctc)
    blank_idx_ctc = char2idx_ctc.get("<BLK>")
    if model_type == "hybrid" and blank_idx_ctc is None:
        raise ValueError("<BLK> token not found in char2idx_ctc. Check data.yaml special_tokens.")

    model_cfg["pad_idx"] = pad_idx_attention


    lr_config_key_log = 'learning_rate_finetune' if model_cfg.get('fine_tune_mode') else 'learning_rate'
    wd_config_key_log = 'Fine_tune_weight_decay' if model_cfg.get('fine_tune_mode') else 'weight_decay'
    hyperparameters_for_log = {
        "batch_size": batch_size,
        "learning_rate": float(model_cfg.get(lr_config_key_log, 1e-4)),
        "weight_decay": float(model_cfg.get(wd_config_key_log, 1e-4)),
        "lambda_ctc": lambda_ctc, "lambda_seg": lambda_seg,
        "u_penalty_weight": u_penalty_weight if use_segmentation_module else 0.0,
        "model_type": model_type, "d_model": d_model,
        "encoder_num_layers": model_cfg.get('encoder_num_layers', "N/A"),
        "decoder_num_layers": model_cfg.get('decoder_num_layers', "N/A"),
        "feature_extractor_type": model_cfg.get('feature_extractor_type', "N/A"),
        "segmentation_loss_type": model_cfg.get('segmentation_loss_type', 'bce') if use_segmentation_module else "N/A",
    }
    if use_segmentation_module and model_cfg.get('segmentation_loss_type', 'bce').lower() == 'focal':
        hyperparameters_for_log["focal_loss_alpha"] = focal_loss_alpha
        hyperparameters_for_log["focal_loss_gamma"] = focal_loss_gamma

    checkpoint_dir_base = model_cfg.get("checkpoint_dir_base", "/home/jiangwenzhu/OCR_result")
    checkpoint_dir = os.path.join(checkpoint_dir_base, "checkpoints")

    experiment_id, training_log_path = initialize_log_file(
        log_dir=checkpoint_dir, hyperparameters=hyperparameters_for_log,
        model_cfg_path=os.path.join(config_base_path, "model.yaml"),
        data_cfg_path=os.path.join(config_base_path, "data.yaml")
    )
    if not training_log_path: return

    try:
        model = EnhancedIndustrialOCR(
            vocab_size=vocab_size_attention, vocab_size_ctc=vocab_size_ctc,
            model_cfg=model_cfg, blank_idx_ctc=blank_idx_ctc
        ).to(device)
    except Exception as e:
        print(f"Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return


    if model_cfg.get('fine_tune_mode', False):
        print("Fine-tuning mode enabled。")
        pretrained_path = model_cfg.get('pretrained_checkpoint_path')
        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location=device)
            model_weights = checkpoint.get('model_state_dict', checkpoint)
            if any(key.startswith('module.') for key in model_weights.keys()):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k_model, v_model in model_weights.items(): new_state_dict[
                    k_model[7:] if k_model.startswith('module.') else k_model] = v_model
                model_weights = new_state_dict


            missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
            print("成功loaded预训练权重。")

            filter_out_prefixes = ['unet_segmentation_head', 'learned_gating_module',
                                   'feature_extractor.unet_segmentation_head',
                                   'feature_extractor.learned_gating_module',
                                   'feature_extractor.adaptive_final_downsampler.final_output_conv',

                                   'ctc_fc', 'fc_out']

            relevant_missing_keys = [k_miss for k_miss in missing_keys if
                                     not any(k_miss.startswith(p_fix) for p_fix in filter_out_prefixes)]
            if relevant_missing_keys: print("Warning: Potentially relevant weights in the model were not found in the checkpoint:", relevant_missing_keys)
            if unexpected_keys: print("Warning: Unexpected weights present in the checkpoint that do not exist in the model.:", unexpected_keys)
        else:
            print(f"Warning: No pre-trained checkpoint found in fine-tuning mode '{pretrained_path}'")

        if model_cfg.get('freeze_feature_extractor', False):
            num_frozen_params = 0
            for name, param in model.named_parameters():
                if name.startswith('feature_extractor.'):
                    param.requires_grad = False
                    num_frozen_params += 1

    model.to(device)

    train_img_dir = data_cfg.get('img_dir_train')
    train_ann_file = data_cfg.get('ann_path_train')
    val_img_dir = data_cfg.get('img_dir_val')
    val_ann_file = data_cfg.get('ann_path_val')


    try:
        transform_train = MeterAugment(is_resize=model_cfg.get("is_resize", False))
        train_set = MeterDataset(
            img_dir=train_img_dir, ann_path=train_ann_file, chars=chars,
            max_len=max_len_attention, special_tokens=special_tokens,
            transform=transform_train, is_grayscale=is_grayscale_dataset,
            data_cfg=data_cfg,model_cfg=model_cfg,grouping_config=None
        )
        val_set = MeterDataset(
            img_dir=val_img_dir, ann_path=val_ann_file, chars=chars,
            max_len=max_len_attention, special_tokens=special_tokens,
            transform=None, is_grayscale=is_grayscale_dataset,
            data_cfg=data_cfg,model_cfg=model_cfg,grouping_config=train_set.grouping_config
        )

        collate_fn_instance = CollateFN(
            batch_size=batch_size, char2idx_ctc=char2idx_ctc,
            max_collate_h=model_cfg.get("max_collate_h", 250),
            max_collate_w=model_cfg.get("max_collate_w", 500),
            target_feature_map_size=(model_cfg.get("feature_map_h"), model_cfg.get("feature_map_w"))
        )

        train_loader = DataLoader(
            train_set, batch_sampler=BatchSampler(RandomSampler(train_set), batch_size, drop_last=True),
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_instance
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_instance
        )

    except Exception as e:

        import traceback
        traceback.print_exc()
        return


    lr_config_key = 'learning_rate_finetune' if model_cfg.get('fine_tune_mode') else 'learning_rate'
    current_learning_rate = float(model_cfg.get(lr_config_key, 1e-4))
    wd_config_key = 'Fine_tune_weight_decay' if model_cfg.get('fine_tune_mode') else 'weight_decay'
    current_weight_decay = float(model_cfg.get(wd_config_key, 1e-4))


    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(params_to_optimize, lr=current_learning_rate, weight_decay=current_weight_decay)

    criterion_attention = CrossEntropyLoss(ignore_index=pad_idx_attention)
    criterion_ctc = None
    if model_type == "hybrid":
        criterion_ctc = CTCLoss(blank=blank_idx_ctc, reduction='mean', zero_infinity=True)


    criterion_seg = None
    if use_segmentation_module:
        seg_loss_type = model_cfg.get('segmentation_loss_type', 'bce').lower()
        if seg_loss_type == 'bce':
            criterion_seg = BCEWithLogitsLoss()

        elif seg_loss_type == 'focal':

            criterion_seg = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma, reduction='mean')

        else:
            raise ValueError(f"未知的Segmentation loss type: {seg_loss_type}")


    scheduler_patience = model_cfg.get("scheduler_patience", 5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)


    use_amp = bool(model_cfg.get("use_amp", False))
    scaler = GradScaler(enabled=use_amp)


    if model_cfg.get('detect_anomaly', False):
        torch.autograd.set_detect_anomaly(True)



    best_val_metric = float('inf')
    epochs_no_improve_count = 0
    best_model_epoch = -1


    num_epochs_key = 'epochs_finetune' if model_cfg.get('fine_tune_mode') else 'epochs'
    num_epochs = model_cfg.get(num_epochs_key, 50)
    accumulation_steps = model_cfg.get("accumulation_steps", 4)
    clip_max_norm = model_cfg.get("clip_max_norm", 1.0)
    eval_frequency = model_cfg.get("eval_frequency", 1)
    early_stopping_patience = model_cfg.get("early_stopping_patience", 10)
    save_prefix = "finetuned_" if model_cfg.get('fine_tune_mode', False) else ""


    for epoch in range(num_epochs):
        model.train()


        epoch_train_total_loss_sum = 0.0
        epoch_train_ctc_loss_sum_accum = 0.0
        epoch_train_attn_loss_sum_accum = 0.0
        epoch_train_seg_loss_combined_sum_accum = 0.0


        count_train_processed_batches = 0
        count_train_ctc_samples_processed = 0
        count_train_attn_samples_processed = 0
        count_train_seg_samples_processed = 0

        optimizer.zero_grad()
        start_time_epoch = time.time()

        for i, batch in enumerate(train_loader):
            if not batch: continue

            src = batch["image"].to(device)
            tgt_input_attn = batch["text_seq"][:, :-1].to(device)
            target_labels_for_attn = batch["text_seq"][:, 1:].to(device)
            src_img_mask = batch["mask"].to(device)
            batch_image_id = batch["image_id"][0]
            gt_seg_mask_train = None
            if use_segmentation_module:
                gt_seg_mask_train = batch['gt_segmentation_mask']
                if gt_seg_mask_train is None: raise ValueError("Split enabled but training data loader not provided 'gt_segmentation_mask'")
                gt_seg_mask_train = gt_seg_mask_train.to(device).float()

            current_batch_s_train = src.shape[0]

            if i == 0 :
                save_dir = "debug_batch_vis"
                os.makedirs(save_dir, exist_ok=True)
                src_cpu = src.detach().cpu()
                mask_cpu = src_img_mask.detach().cpu()

                for i in range(src_cpu.size(0)):
                    img = src_cpu[i]
                    mask = mask_cpu[i]

                    img_min = img.min()
                    img_max = img.max()
                    img = (img - img_min) / (img_max - img_min + 1e-6)

                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)

                    vutils.save_image(
                        img,
                        os.path.join(save_dir, f"batch0_img_{i}.png")
                    )

                    vutils.save_image(
                        mask.float(),
                        os.path.join(save_dir, f"batch0_mask_{i}.png")
                    )

                    overlay = img * mask
                    vutils.save_image(
                        overlay,
                        os.path.join(save_dir, f"batch0_overlay_{i}.png")
                    )

            with autocast(enabled=use_amp):

                model_outputs = model(src, tgt_input_attn, src_img_mask=src_img_mask, batch_image_id = batch_image_id,return_visualizations=False)
                ctc_output_log_probs, output_logits_attn, seg_logits_s2_for_loss, ctc_input_lengths_from_model = \
                    model_outputs[0], model_outputs[1], model_outputs[2], model_outputs[3]

                current_step_total_loss = torch.tensor(0.0, device=device)

                # --- Attention Loss ---
                loss_attention_step = torch.tensor(0.0, device=device)
                if not (model_type == "hybrid" and lambda_ctc == 1.0 and lambda_seg == 0.0):
                    output_logits_permuted = output_logits_attn.permute(1, 0, 2).contiguous()
                    output_logits_flat = output_logits_permuted.reshape(-1, output_logits_permuted.size(-1))
                    target_labels_flat = target_labels_for_attn.reshape(-1)
                    loss_attention_calc = criterion_attention(output_logits_flat, target_labels_flat)
                    if not (torch.isnan(loss_attention_calc) or torch.isinf(loss_attention_calc)):
                        loss_attention_step = loss_attention_calc
                        current_step_total_loss += (1.0 - lambda_ctc - lambda_seg) * loss_attention_step
                        epoch_train_attn_loss_sum_accum += loss_attention_step.item() * current_batch_s_train
                        count_train_attn_samples_processed += current_batch_s_train

                # --- CTC Loss (with filtering) ---
                loss_ctc_step = torch.tensor(0.0, device=device)
                num_valid_for_ctc_this_step = 0
                if model_type == "hybrid" and ctc_output_log_probs is not None and criterion_ctc is not None:
                    ctc_targets_b = batch["ctc_targets"].to(device)
                    ctc_target_lengths_b = batch["ctc_target_lengths"].to(device)
                    ctc_input_lengths_b = ctc_input_lengths_from_model.to(device)
                    original_bs_ctc_step = ctc_output_log_probs.shape[1]  # Batch size for CTC

                    valid_idx_c1 = ctc_input_lengths_b > 0
                    valid_idx_c2 = ctc_input_lengths_b >= ctc_target_lengths_b
                    valid_idx_ctc = valid_idx_c1 & valid_idx_c2
                    num_valid_for_ctc_this_step = valid_idx_ctc.sum().item()

                    if num_valid_for_ctc_this_step > 0:
                        ctc_out_logs_f, tgts_f, in_lens_f, tgt_lens_f = \
                            ctc_output_log_probs, ctc_targets_b, ctc_input_lengths_b, ctc_target_lengths_b
                        if num_valid_for_ctc_this_step < original_bs_ctc_step:

                            ctc_out_logs_f = ctc_output_log_probs[:, valid_idx_ctc, :]
                            in_lens_f = ctc_input_lengths_b[valid_idx_ctc]
                            tgt_lens_f = ctc_target_lengths_b[valid_idx_ctc]

                            filt_tgts_list = []
                            offset = 0
                            for k_train in range(original_bs_ctc_step):
                                length_k = ctc_target_lengths_b[k_train].item()
                                if valid_idx_ctc[k_train]:
                                    filt_tgts_list.append(
                                    ctc_targets_b[offset: offset + length_k])
                                offset += length_k
                            if filt_tgts_list:
                                tgts_f = torch.cat(filt_tgts_list)
                            else:
                                tgts_f = torch.empty(0, dtype=torch.long,
                                                     device=device)

                        if tgts_f.numel() == tgt_lens_f.sum().item():
                            try:
                                loss_ctc_calc_train = criterion_ctc(ctc_out_logs_f, tgts_f, in_lens_f, tgt_lens_f)
                                if not (torch.isnan(loss_ctc_calc_train) or torch.isinf(loss_ctc_calc_train)):
                                    loss_ctc_step = loss_ctc_calc_train
                            except RuntimeError as e_ctc_rt_train:
                                print(f"ERROR Train CTC (Epoch {epoch + 1}, Step {i + 1}): {e_ctc_rt_train}")


                    if num_valid_for_ctc_this_step > 0 and not (
                            torch.isnan(loss_ctc_step) or torch.isinf(loss_ctc_step)):

                        current_step_total_loss += lambda_ctc * loss_ctc_step
                        epoch_train_ctc_loss_sum_accum += loss_ctc_step.item() * num_valid_for_ctc_this_step
                        count_train_ctc_samples_processed += num_valid_for_ctc_this_step

                # --- Segmentation Loss (with U-Penalty) ---
                loss_seg_combined_step = torch.tensor(0.0, device=device)
                if use_segmentation_module and seg_logits_s2_for_loss is not None and criterion_seg is not None:
                    loss_seg_main_step = criterion_seg(seg_logits_s2_for_loss, gt_seg_mask_train)
                    loss_seg_u_penalty_step = model.u_shaped_segmentation_penalty(
                        seg_logits_s2_for_loss,
                        target_min_proportion=u_penalty_target_min,
                        target_max_proportion=u_penalty_target_max,
                        penalty_weight=u_penalty_weight
                    )
                    loss_seg_combined_calc = loss_seg_main_step + loss_seg_u_penalty_step

                    if not (torch.isnan(loss_seg_combined_calc) or torch.isinf(loss_seg_combined_calc)):
                        loss_seg_combined_step = loss_seg_combined_calc
                        current_step_total_loss += lambda_seg * loss_seg_combined_step
                        epoch_train_seg_loss_combined_sum_accum += loss_seg_combined_step.item() * current_batch_s_train
                        count_train_seg_samples_processed += current_batch_s_train

                loss_for_backward = current_step_total_loss / accumulation_steps

            scaler.scale(loss_for_backward).backward()
            epoch_train_total_loss_sum += current_step_total_loss.item()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                count_train_processed_batches += 1  #

                if count_train_processed_batches > 0 and count_train_processed_batches % 10 == 0:
                    current_lr_print = optimizer.param_groups[0]['lr']
                    avg_total_loss_print = epoch_train_total_loss_sum / count_train_processed_batches  # Avg of batch mean total losses
                    avg_ctc_loss_print = epoch_train_ctc_loss_sum_accum / count_train_ctc_samples_processed if count_train_ctc_samples_processed > 0 else 0
                    avg_attn_loss_print = epoch_train_attn_loss_sum_accum / count_train_attn_samples_processed if count_train_attn_samples_processed > 0 else 0
                    avg_seg_loss_print = epoch_train_seg_loss_combined_sum_accum / count_train_seg_samples_processed if count_train_seg_samples_processed > 0 else 0

                    print(
                        f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], OptStep [{count_train_processed_batches}], LR: {current_lr_print:.2e}, "
                        f"Batch Avg Total Loss: {avg_total_loss_print:.4f} "
                        f"(CTC: {avg_ctc_loss_print:.4f}, Attn: {avg_attn_loss_print:.4f}, SegCombined: {avg_seg_loss_print:.4f})")


        avg_epoch_train_loss_total = epoch_train_total_loss_sum / count_train_processed_batches if count_train_processed_batches > 0 else float(
            'nan')
        avg_epoch_train_loss_ctc = epoch_train_ctc_loss_sum_accum / count_train_ctc_samples_processed if count_train_ctc_samples_processed > 0 else float(
            'nan')
        avg_epoch_train_loss_attention = epoch_train_attn_loss_sum_accum / count_train_attn_samples_processed if count_train_attn_samples_processed > 0 else float(
            'nan')
        avg_epoch_train_loss_seg = epoch_train_seg_loss_combined_sum_accum / count_train_seg_samples_processed if count_train_seg_samples_processed > 0 else float(
            'nan')
        epoch_duration = time.time() - start_time_epoch
        print(f"--- Epoch {epoch + 1} Training completed,  Time elapsed: {epoch_duration:.2f}s ---")
        print(f"    Avg Train Total Loss (avg over opt_steps): {avg_epoch_train_loss_total:.4f}")
        if model_type == "hybrid": print(
            f"    Avg Train CTC Loss   : {avg_epoch_train_loss_ctc:.4f} (avg per sample, on {count_train_ctc_samples_processed} samples)")
        if not (model_type == "hybrid" and lambda_ctc == 1.0 and lambda_seg == 0.0): print(
            f"    Avg Train Attn Loss  : {avg_epoch_train_loss_attention:.4f} (avg per sample, on {count_train_attn_samples_processed} samples)")
        if use_segmentation_module: print(
            f"    Avg Train SegCombined Loss : {avg_epoch_train_loss_seg:.4f} (avg per sample, on {count_train_seg_samples_processed} samples)")


        if (epoch + 1) % eval_frequency == 0:
            model.eval()

            epoch_val_total_loss_sum_accum = 0.0
            epoch_val_ctc_loss_sum_accum = 0.0
            epoch_val_attn_loss_sum_accum = 0.0
            epoch_val_seg_loss_sum_accum = 0.0

            count_val_processed_batches = 0
            total_val_samples_overall = 0
            total_val_samples_ctc_processed = 0
            total_val_samples_attn_valid = 0
            total_val_samples_seg_valid = 0

            total_edit_distance = 0
            total_target_length_for_cer = 0
            total_correct_sequences = 0

            start_time_val = time.time()
            visualized_this_epoch_already = False

            with torch.no_grad():
                for val_batch_idx, batch_val in enumerate(val_loader):
                    if not batch_val: continue

                    src = batch_val["image"].to(device)
                    tgt_for_loss_input = batch_val["text_seq"][:, :-1].to(device)
                    target_for_attn_loss = batch_val["text_seq"][:, 1:].to(device)
                    src_img_mask_val = batch_val["mask"].to(device)
                    ground_truth_labels_val = batch_val["label"]
                    batch_image_id = batch_val["image_id"]
                    gt_seg_mask_val = None
                    if use_segmentation_module:
                        gt_seg_mask_val = batch_val["gt_segmentation_mask"]
                        if gt_seg_mask_val is None: raise ValueError("Val: gt_segmentation_mask missing.")
                        gt_seg_mask_val = gt_seg_mask_val.to(device).float()

                    current_batch_s_val = src.shape[0]
                    total_val_samples_overall += current_batch_s_val

                    should_visualize_this_batch_val = (
                                val_batch_idx == 0 and pce_vis and epoch == epoch_to_visualize_val and not visualized_this_epoch_already)

                    model_outputs_val_tuple = model(
                        src, tgt_for_loss_input, src_img_mask=src_img_mask_val, batch_image_id = batch_image_id,
                        return_visualizations=should_visualize_this_batch_val
                    )


                    if should_visualize_this_batch_val:
                        ctc_log_probs_val, output_logits_val, src_viz, x_feat_viz, cpe_viz, \
                            x_pe_viz, seg_logits_s2_val, ctc_input_lengths_val_from_model = model_outputs_val_tuple

                        current_image_id_val = batch_val["image_id"][0]
                        viz_parent_dir_exp = os.path.join(checkpoint_dir, "visualizations", experiment_id)
                        current_epoch_viz_dir_val = os.path.join(viz_parent_dir_exp, f"epoch_{epoch + 1}")
                        if src_viz.ndim == 4 and src_viz.shape[0] > 0:  # Check if batch and not empty
                            save_feature_visualization(
                                epoch=epoch + 1, batch_image_id=str(current_image_id_val),
                                original_src=src_viz, x_feat=x_feat_viz, cpe=cpe_viz, x_cpe=x_pe_viz,
                                output_dir=current_epoch_viz_dir_val, sample_idx_in_batch=0
                            )
                            print(f"Epoch {epoch + 1} The visual image has been saved to {current_epoch_viz_dir_val}")
                        visualized_this_epoch_already = True
                    else:
                        ctc_log_probs_val, output_logits_val, seg_logits_s2_val, ctc_input_lengths_val_from_model = model_outputs_val_tuple

                    val_step_total_loss = torch.tensor(0.0, device=device)

                    loss_att_val_step = torch.tensor(0.0, device=device)
                    if not (model_type == "hybrid" and lambda_ctc == 1.0 and lambda_seg == 0.0):
                        output_val_permuted = output_logits_val.permute(1, 0, 2).contiguous()
                        output_logits_flat_val = output_val_permuted.reshape(-1, output_val_permuted.size(-1))
                        target_labels_flat_val = target_for_attn_loss.reshape(-1)
                        loss_att_val_calc = criterion_attention(output_logits_flat_val, target_labels_flat_val)
                        if not (torch.isnan(loss_att_val_calc) or torch.isinf(loss_att_val_calc)):
                            loss_att_val_step = loss_att_val_calc
                            val_step_total_loss += (1.0 - lambda_ctc - lambda_seg) * loss_att_val_step
                            epoch_val_attn_loss_sum_accum += loss_att_val_step.item() * current_batch_s_val
                            total_val_samples_attn_valid += current_batch_s_val


                    loss_ctc_val_step = torch.tensor(0.0, device=device)
                    num_valid_for_ctc_val_step = 0
                    if model_type == "hybrid" and ctc_log_probs_val is not None and criterion_ctc is not None:
                        ctc_targets_val_b = batch_val["ctc_targets"].to(device)
                        ctc_target_lengths_val_b = batch_val["ctc_target_lengths"].to(device)
                        ctc_input_lengths_val_b = ctc_input_lengths_val_from_model.to(device)
                        original_bs_ctc_val_step = ctc_log_probs_val.shape[1]

                        valid_idx_c1_val = ctc_input_lengths_val_b > 0
                        valid_idx_c2_val = ctc_input_lengths_val_b >= ctc_target_lengths_val_b
                        valid_idx_ctc_val = valid_idx_c1_val & valid_idx_c2_val
                        num_valid_for_ctc_val_step = valid_idx_ctc_val.sum().item()

                        if num_valid_for_ctc_val_step > 0:
                            ctc_out_logs_f_val, tgts_f_val, in_lens_f_val, tgt_lens_f_val = \
                                ctc_log_probs_val, ctc_targets_val_b, ctc_input_lengths_val_b, ctc_target_lengths_val_b
                            if num_valid_for_ctc_val_step < original_bs_ctc_val_step:
                                ctc_out_logs_f_val = ctc_log_probs_val[:, valid_idx_ctc_val, :]
                                in_lens_f_val = ctc_input_lengths_val_b[valid_idx_ctc_val]
                                tgt_lens_f_val = ctc_target_lengths_val_b[valid_idx_ctc_val]
                                filt_tgts_list_val = []
                                offset_val = 0
                                for k_val in range(original_bs_ctc_val_step):
                                    l_val = ctc_target_lengths_val_b[k_val].item()
                                    if valid_idx_ctc_val[k_val]: filt_tgts_list_val.append(
                                        ctc_targets_val_b[offset_val: offset_val + l_val])
                                    offset_val += l_val
                                if filt_tgts_list_val:
                                    tgts_f_val = torch.cat(filt_tgts_list_val)
                                else:
                                    tgts_f_val = torch.empty(0, dtype=torch.long, device=device)

                            if tgts_f_val.numel() == tgt_lens_f_val.sum().item():
                                try:
                                    loss_ctc_val_calc = criterion_ctc(ctc_out_logs_f_val, tgts_f_val, in_lens_f_val,
                                                                      tgt_lens_f_val)
                                    if not (torch.isnan(loss_ctc_val_calc) or torch.isinf(loss_ctc_val_calc)):
                                        loss_ctc_val_step = loss_ctc_val_calc
                                except RuntimeError as e_ctc_val_rt_inner_val:
                                    print(f"ERROR val CTC inner: {e_ctc_val_rt_inner_val}")

                        if num_valid_for_ctc_val_step > 0 and not (
                                torch.isnan(loss_ctc_val_step) or torch.isinf(loss_ctc_val_step)):
                            if not (
                                    loss_ctc_val_step.item() == 0.0 and num_valid_for_ctc_val_step > 0 and not torch.all(
                                    valid_idx_ctc_val)):
                                val_step_total_loss += lambda_ctc * loss_ctc_val_step
                            epoch_val_ctc_loss_sum_accum += loss_ctc_val_step.item() * num_valid_for_ctc_val_step
                            total_val_samples_ctc_processed += num_valid_for_ctc_val_step


                    loss_seg_val_combined_step = torch.tensor(0.0, device=device)
                    if use_segmentation_module and seg_logits_s2_val is not None and criterion_seg is not None:
                        loss_seg_main_val = criterion_seg(seg_logits_s2_val, gt_seg_mask_val)
                        loss_seg_u_penalty_val = model.u_shaped_segmentation_penalty(
                            seg_logits_s2_val, target_min_proportion=u_penalty_target_min,
                            target_max_proportion=u_penalty_target_max, penalty_weight=u_penalty_weight
                        )
                        loss_seg_val_combined_calc = loss_seg_main_val + loss_seg_u_penalty_val
                        if not (torch.isnan(loss_seg_val_combined_calc) or torch.isinf(loss_seg_val_combined_calc)):
                            loss_seg_val_combined_step = loss_seg_val_combined_calc
                            val_step_total_loss += lambda_seg * loss_seg_val_combined_step
                            epoch_val_seg_loss_sum_accum += loss_seg_val_combined_step.item() * current_batch_s_val
                            total_val_samples_seg_valid += current_batch_s_val

                    epoch_val_total_loss_sum_accum += val_step_total_loss.item()
                    count_val_processed_batches += 1

                    try:
                        predicted_indices = model.predict(src, src_img_mask_val, max_len_attention,
                                                          sos_idx_attention, eos_idx_attention, pad_idx_attention)
                        predicted_labels_val_list = []
                        for idx_sequence_val in predicted_indices:
                            pred_chars_val = []
                            for idx_tensor_val in idx_sequence_val:
                                idx_val = idx_tensor_val.item()
                                if idx_val == eos_idx_attention: break
                                if idx_val == pad_idx_attention or idx_val == sos_idx_attention: continue
                                pred_chars_val.append(idx2char_attention.get(idx_val, '?'))
                            predicted_labels_val_list.append("".join(pred_chars_val))

                        for gt_val, pred_str_val in zip(ground_truth_labels_val, predicted_labels_val_list):
                            if pred_str_val == gt_val: total_correct_sequences += 1
                            if len(gt_val) > 0:
                                total_edit_distance += editdistance.eval(pred_str_val, gt_val)
                                total_target_length_for_cer += len(gt_val)
                    except Exception as predict_e_val:
                        print(f"Val predict/CER error: {predict_e_val}")


            avg_val_loss_total = epoch_val_total_loss_sum_accum / count_val_processed_batches if count_val_processed_batches > 0 else float(
                'nan')
            avg_val_loss_ctc = epoch_val_ctc_loss_sum_accum / total_val_samples_ctc_processed if total_val_samples_ctc_processed > 0 else float(
                'nan')
            avg_val_loss_attention = epoch_val_attn_loss_sum_accum / total_val_samples_attn_valid if total_val_samples_attn_valid > 0 else float(
                'nan')
            avg_val_loss_seg = epoch_val_seg_loss_sum_accum / total_val_samples_seg_valid if total_val_samples_seg_valid > 0 else float(
                'nan')
            val_cer = (total_edit_distance / total_target_length_for_cer) * 100 if total_target_length_for_cer > 0 else float(
                'inf')
            val_acc = (total_correct_sequences / total_val_samples_overall) * 100 if total_val_samples_overall > 0 else 0.0
            val_duration = time.time() - start_time_val

            print(f"--- Epoch {epoch + 1} Validation completed ---")
            print(f"    Avg Val Total Loss (avg over batches): {avg_val_loss_total:.4f}")
            if model_type == "hybrid": print(
                f"    Avg Val CTC Loss   : {avg_val_loss_ctc:.4f} (avg per sample, on {total_val_samples_ctc_processed} samples)")
            if not (model_type == "hybrid" and lambda_ctc == 1.0 and lambda_seg == 0.0): print(
                f"    Avg Val Attn Loss  : {avg_val_loss_attention:.4f} (avg per sample, on {total_val_samples_attn_valid} samples)")
            if use_segmentation_module: print(
                f"    Avg Val Seg Loss   : {avg_val_loss_seg:.4f} (avg per sample, on {total_val_samples_seg_valid} samples)")
            print(f"    CER (%)            : {val_cer:.2f}")
            print(f"    Sequence Acc (%)   : {val_acc:.2f}")
            print(f"    Time elapsed: {val_duration:.2f}s")

            epoch_log_data = {
                "epoch": epoch + 1,
                "train_loss_total": avg_epoch_train_loss_total, "train_loss_ctc": avg_epoch_train_loss_ctc,
                "train_loss_attention": avg_epoch_train_loss_attention, "train_loss_seg": avg_epoch_train_loss_seg,
                "val_loss_total": avg_val_loss_total, "val_loss_ctc": avg_val_loss_ctc,
                "val_loss_attention": avg_val_loss_attention, "avg_val_loss_seg": avg_val_loss_seg,
                "val_cer": val_cer, "val_sequence_acc": val_acc
            }
            log_epoch_data(training_log_path, experiment_id, epoch_log_data, hyperparameters_for_log)
            scheduler.step(val_cer)

            current_val_metric = val_cer
            if current_val_metric < best_val_metric:
                print(f" Discover the new best Validation CER: {current_val_metric:.2f}% (better than {best_val_metric:.2f}%)")
                best_val_metric = current_val_metric
                best_epoch = epoch + 1
                epochs_no_improve_count = 0
                best_model_filename = f"{save_prefix}best_model_epoch{best_epoch}_cer_{best_val_metric:.2f}.pth"
                current_best_model_path = os.path.join(checkpoint_dir, best_model_filename)
                print(f"    Saving best model to {current_best_model_path}")
                try:
                    torch.save({
                        'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'best_val_cer': best_val_metric,

                        'scaler_state_dict': scaler.state_dict() if use_amp else None,
                    }, current_best_model_path)
                except Exception as e_save:
                    print(f"!!! Error: Failed to save best model: {e_save}")
            else:
                epochs_no_improve_count += eval_frequency

            model.train()
            if epochs_no_improve_count >= early_stopping_patience:
                print(
                    f"\n!!! Early Stop Trigger：Validation CER  {epochs_no_improve_count} epochs without improvement. Stopping at Epoch {epoch + 1} stopping training。")
                break

    print("Training (fine-tuning) completed!")
    if best_epoch != -1:
        print(f"{save_prefix}Best model saved at Epoch {best_epoch} (Based on the minimum Validation CER: {best_val_metric:.2f}%)")
    else:
        print("No optimal model meeting the specified criteria was found during training.（")


if __name__ == '__main__':
    main()
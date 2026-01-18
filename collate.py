# OCR/utils/collate.py
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class CollateFN(object):
    def __init__(self, batch_size,char2idx_ctc='', max_collate_h=650, max_collate_w=800,target_feature_map_size=(16, 32)):
        self.batch_size = batch_size
        self.MAX_COLLATE_HEIGHT = max_collate_h
        self.MAX_COLLATE_WIDTH = max_collate_w

        self.char2idx_ctc = char2idx_ctc
        self.target_feature_map_size = target_feature_map_size


    def __call__(self, batch_list_from_dataset):
        processed_items = []
        if not batch_list_from_dataset:

            return self._get_empty_batch(self.batch_size, 19, torch.device('cpu'))


        for item_idx, item in enumerate(batch_list_from_dataset):
            img_roi = item.get('image')
            img_id = item.get('image_id', f'idx_{item_idx}')

            valid = True
            if not isinstance(img_roi, torch.Tensor) or img_roi.ndim != 3 or img_roi.shape[0] not in [1, 3]:
                print(
                    f"Warning: CollateFN - sample {img_id} image invalid ({type(img_roi)}, shape={img_roi.shape if hasattr(img_roi, 'shape') else 'N/A'})，skip。")
                valid = False
            elif img_roi.shape[1] <= 0 or img_roi.shape[2] <= 0:
                print(f"Warning: CollateFN - invalid image dimensions ({img_roi.shape[1]}x{img_roi.shape[2]}) into {img_id} ，skip。")
                valid = False



            if not valid: continue

            c, h_roi, w_roi = img_roi.shape
            current_img_for_padding = img_roi
            current_h, current_w = h_roi, w_roi


            if current_h > self.MAX_COLLATE_HEIGHT or current_w > self.MAX_COLLATE_WIDTH:
                scale = min(self.MAX_COLLATE_HEIGHT / current_h, self.MAX_COLLATE_WIDTH / current_w)

                new_h = max(1, int(round(current_h * scale)))
                new_w = max(1, int(round(current_w * scale)))

                try:
                    interpolation_mode = 'area' if scale < 1.0 else 'bilinear'

                    align_corners_setting = None if interpolation_mode == 'area' else False

                    current_img_for_padding_unsqueezed = current_img_for_padding.unsqueeze(0)
                    current_img_for_padding_resized = F.interpolate(
                        current_img_for_padding_unsqueezed, size=(new_h, new_w),
                        mode=interpolation_mode, align_corners=align_corners_setting
                    )
                    current_img_for_padding = current_img_for_padding_resized.squeeze(0)
                    current_h, current_w = new_h, new_w
                except Exception as resize_e:
                    print(f"!!! Error: F.interpolate failed during Collate, sample {img_id}，skip: {resize_e}")
                    continue

            temp_item = item.copy()
            temp_item['image_to_pad'] = current_img_for_padding

            temp_item['bbox_for_norm'] = [0.0, 0.0, float(current_w), float(current_h)]
            temp_item['current_shape_for_padding'] = (current_h, current_w)
            processed_items.append(temp_item)

        if not processed_items:

            example_item_for_empty = batch_list_from_dataset[0] if batch_list_from_dataset else {}

            text_len_for_empty = \
            example_item_for_empty.get('text_seq', torch.zeros(item.get('max_len', 19) if item else 19)).shape[0]

            device_for_empty = torch.device('cpu')
            if 'img_roi' in locals() and isinstance(img_roi, torch.Tensor):
                device_for_empty = img_roi.device
            elif batch_list_from_dataset and isinstance(batch_list_from_dataset[0].get('image'), torch.Tensor):
                device_for_empty = batch_list_from_dataset[0].get('image').device


            return self._get_empty_batch(self.batch_size, text_len_for_empty, device_for_empty)


        max_h = max(pi['image_to_pad'].shape[1] for pi in processed_items)
        max_w = max(pi['image_to_pad'].shape[2] for pi in processed_items)


        padded_images = []
        adjusted_bboxes_normalized = []
        padding_masks = []
        batch_gt_seg_mask = None

        text_seqs = [p_item['text_seq'] for p_item in processed_items]
        labels = [p_item['label'] for p_item in processed_items]

        ctc_targets_list = []
        ctc_target_lengths_list = []


        for item in processed_items:
            label_str = item['label']
            current_ctc_indices_for_sample = []

            for char_val in label_str:

                if char_val in self.char2idx_ctc:
                    char_idx = self.char2idx_ctc[char_val]

                    current_ctc_indices_for_sample.append(char_idx)
                else:

                    print(f"Warning: character '{char_val}' in label '{label_str}' not found in char2idx_ctc. Will be ignored。")

            ctc_target_lengths_list.append(len(current_ctc_indices_for_sample))
            ctc_targets_list.extend(current_ctc_indices_for_sample)

        batch_ctc_targets_tensor = torch.tensor(ctc_targets_list, dtype=torch.long)
        ctc_target_lengths_list = torch.tensor(ctc_target_lengths_list, dtype=torch.long)
        image_ids = [p_item.get('image_id', -1) for p_item in processed_items]
        areas = []
        padded_gt_segmentation_masks = []

        for p_item in processed_items:
            img_to_pad = p_item['image_to_pad']
            h_content, w_content = p_item['current_shape_for_padding']


            pad_h_needed = max_h - h_content
            pad_w_needed = max_w - w_content


            padded_img = F.pad(img_to_pad, (0, pad_w_needed, 0, pad_h_needed), value=0.0)
            padded_images.append(padded_img)


            norm_x1 = 0.0 / max_w
            norm_y1 = 0.0 / max_h
            norm_x2 = float(w_content) / max_w
            norm_y2 = float(h_content) / max_h

            adjusted_bbox_norm = [
                max(0.0, min(1.0, norm_x1)), max(0.0, min(1.0, norm_y1)),
                max(0.0, min(1.0, norm_x2)), max(0.0, min(1.0, norm_y2))
            ]
            adjusted_bboxes_normalized.append(adjusted_bbox_norm)

            mask = torch.zeros((1, max_h, max_w), dtype=torch.bool, device=img_to_pad.device)
            mask[:, :h_content, :w_content] = True
            padding_masks.append(mask)


            area_val = p_item.get('area', 0.0)
            if not isinstance(area_val, torch.Tensor):
                area_val = torch.tensor([float(area_val)], dtype=torch.float32, device=img_to_pad.device)
            elif area_val.ndim == 0:
                area_val = area_val.unsqueeze(0)
            areas.append(area_val.to(img_to_pad.device))

            gt_seg_mask = None
            gt_seg_mask_tensor = p_item.get("gt_segmentation_mask")

            if isinstance(gt_seg_mask_tensor, np.ndarray):
                gt_seg_mask = torch.from_numpy(gt_seg_mask_tensor).unsqueeze(0)
            elif torch.is_tensor(gt_seg_mask_tensor):
                gt_seg_mask = gt_seg_mask_tensor.unsqueeze(0)
            else:
                print(f"Warning: unexpected type of gt_segmentation_mask：{type(gt_seg_mask_tensor)}")


            if gt_seg_mask is not None:
                h_content, w_content = p_item['current_shape_for_padding']
                pad_h_needed = max_h - h_content
                pad_w_needed = max_w - w_content


                padded_gt_seg_mask_batch_res = F.pad(gt_seg_mask, (0, pad_w_needed, 0, pad_h_needed),
                                                     value=0.0)

                if self.target_feature_map_size is not None:

                    resized_gt_seg_mask = F.interpolate(
                        padded_gt_seg_mask_batch_res.unsqueeze(0),
                        size=self.target_feature_map_size,
                        mode='nearest'
                    ).squeeze(0)

                    padded_gt_segmentation_masks.append(resized_gt_seg_mask)
                else:
                    padded_gt_segmentation_masks.append(padded_gt_seg_mask_batch_res)

        try:
            batch_image = torch.stack(padded_images)
            batch_padding_mask = torch.stack(padding_masks)
            batch_text_seq = torch.stack(text_seqs)
            batch_area = torch.stack(areas)
            batch_ad_bbox_norm = torch.tensor(adjusted_bboxes_normalized, dtype=torch.float32).to(batch_image.device)
            if padded_gt_segmentation_masks:
                batch_gt_seg_mask = torch.stack(padded_gt_segmentation_masks)
        except Exception as stack_e:
            print(f"!!! Error: Error when stacking batch: {stack_e}")
            example_item_for_empty_stack = processed_items[0] if processed_items else {}
            text_len_for_empty_stack = \
            example_item_for_empty_stack.get('text_seq', torch.zeros(item.get('max_len', 19) if item else 19)).shape[0]
            device_for_empty_stack = torch.device('cpu')
            if 'img_to_pad' in locals() and isinstance(img_to_pad, torch.Tensor):
                device_for_empty_stack = img_to_pad.device
            elif processed_items and isinstance(processed_items[0].get('image_to_pad'), torch.Tensor):
                device_for_empty_stack = processed_items[0].get('image_to_pad').device

            return self._get_empty_batch(len(processed_items) if processed_items else self.batch_size,
                                         text_len_for_empty_stack,
                                         device_for_empty_stack)

        batch_dict = {
            'image': batch_image,
            'mask': batch_padding_mask,
            'text_seq': batch_text_seq,
            'label': labels,
            'area': batch_area,
            'ad_bbox': batch_ad_bbox_norm,
            'image_id': image_ids,
            'ctc_targets': batch_ctc_targets_tensor,
            'ctc_target_lengths': ctc_target_lengths_list
        }
        if batch_gt_seg_mask is not None:
            batch_dict['gt_segmentation_mask'] = batch_gt_seg_mask
        return batch_dict

    def _get_empty_batch(self, current_batch_size, text_len, device):


        num_channels = 3

        return {
            'image': torch.zeros((current_batch_size, num_channels, 16, 32), dtype=torch.float32, device=device),

            'mask': torch.zeros((current_batch_size, 1, 16, 32), dtype=torch.bool, device=device),
            'text_seq': torch.zeros((current_batch_size, text_len), dtype=torch.long, device=device),
            'label': [""] * current_batch_size,
            'area': torch.zeros((current_batch_size, 1), dtype=torch.float32, device=device),
            'ad_bbox': torch.zeros((current_batch_size, 4), dtype=torch.float32, device=device),
            'image_id': [-1] * current_batch_size,
            'ctc_targets' : torch.empty(0, dtype=torch.long, device=device),
            'ctc_target_lengths' : torch.zeros(current_batch_size, dtype=torch.long, device=device)
        }
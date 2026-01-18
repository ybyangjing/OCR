import json
import os
import pickle
from pathlib import Path
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO  # Ensure pycocotools is installed: pip install pycocotools
from pycocotools import mask as coco_mask_util  # For handling COCO segmentation

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def bbox_to_segmentation(bbox):
    """
    Convert bbox to segmentation format (four vertices)
    bbox: [xmin, ymin, width, height]
    Returns: [[x1, y1, x2, y2, x3, y3, x4, y4]] - clockwise order
    """
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height


    segmentation = [
        xmin, ymin,
        xmax, ymin,
        xmax, ymax,
        xmin, ymax
    ]

    return [segmentation]


class MeterDataset(Dataset):
    def __init__(self, img_dir, ann_path, chars, max_len,
                 model_cfg=None, data_cfg=None, special_tokens=None,
                 transform=None, is_grayscale=False, grouping_config=None):
        """
        Args:
            img_dir (str): Image directory path。
            ann_path (str): Path to COCO format annotation file (JSON)。
            chars (str): Allowed character set。
            max_len (int): Maximum text length (including special tokens)。
            model_cfg (dict): Model configuration dictionary。
            data_cfg (dict): Data configuration dictionary, includes ROI cropping related config。
            special_tokens (list, optional): Special token list。
            transform (callable, optional): Data augmentation applied to images。
            is_grayscale (bool): Whether input images should be processed as grayscale。False means process as 3-channel。
            grouping_config (dict, optional): Grouping configuration。
        """
        self.img_dir = Path(img_dir)
        self.ann_path = ann_path
        self.transform = transform
        self.max_len = max_len  # This is the total length including SOS/EOS, used for Attention
        self.is_grayscale = is_grayscale
        self.data_cfg = data_cfg if data_cfg else {}
        self.model_cfg = model_cfg if model_cfg else {}

        self.use_roi_crop = self.data_cfg.get('use_roi_crop', False)
        self.roi_crop_padding = self.data_cfg.get('roi_crop_padding', 0.1)
        self.roi_min_size = self.data_cfg.get('roi_min_size', 16)

        if self.use_roi_crop:
            print(f"★ ROI cropping mode enabled:")
            print(f"  - Boundary expansion ratio: {self.roi_crop_padding * 100:.0f}%")
            print(f"  - ROI minimum size: {self.roi_min_size}px")

        all_chars_list = list(chars)
        self.char2idx = {char: i for i, char in enumerate(all_chars_list)}
        current_idx = len(all_chars_list)
        if special_tokens:
            for token in special_tokens:
                if token not in self.char2idx:
                    self.char2idx[token] = current_idx
                    current_idx += 1
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        try:
            self.coco = COCO(ann_path)
            self.img_ids = sorted(list(self.coco.imgs.keys()))
            self.samples = self._load_samples()
            if not self.samples:
                raise ValueError("Failed to load any valid samples from annotation file. Please check annotation file and image paths.")
            self.grouping_config = grouping_config
            if self.grouping_config is None and self.model_cfg.get('use_aspect_ratio_grouping', True):
                self._initialize_grouping_config(ann_path)
        except Exception as e:
            print(f"Dataset initialization failed: {e}")
            raise

    def _load_samples(self):
        samples = []
        for ann_id in self.coco.anns:
            ann = self.coco.anns[ann_id]

            img_info = self.coco.loadImgs(ann['image_id'])[0]
            img_w, img_h = img_info['width'], img_info['height']
            segmentation_bbox = []

            x, y, w, h = ann['bbox']
            segmentation_bbox = ann['bbox']
            bbox = [
                max(0, int(x)),
                max(0, int(y)),
                min(img_w - 1, int(x + w)),
                min(img_h - 1, int(y + h))
            ]
            label = ann['attributes']['text']


            segmentation_coords = bbox_to_segmentation(segmentation_bbox)
            binary_mask_orig_res = np.zeros((img_h, img_w), dtype=np.uint8)
            if segmentation_coords:
                try:
                    if not isinstance(segmentation_coords, list) or not all(
                            isinstance(p, list) for p in segmentation_coords):
                        print(
                            f"Warning: image {ann['image_id']} annotation {ann_id} the split format is incorrect; it should be a list of polygons. Mask generation skipped.")
                    else:
                        rles = coco_mask_util.frPyObjects(segmentation_coords, img_h, img_w)
                        mask_decoded = coco_mask_util.decode(rles)
                        if mask_decoded.ndim > 2:
                            binary_mask_orig_res = np.sum(mask_decoded, axis=2, dtype=np.uint8)
                        else:
                            binary_mask_orig_res = mask_decoded.astype(np.uint8)
                        binary_mask_orig_res = np.clip(binary_mask_orig_res, 0, 1)
                except Exception as e:
                    print(f"Warning: Unable to generate segmentation mask for image {ann['image_id']} annotation {ann_id} segmentation mask: {e}")

            samples.append({
                "image_id": ann['image_id'],
                "img_path": str(self.img_dir / img_info['file_name']),
                "file_name": img_info['file_name'],
                "area": ann['area'],
                "label": label,
                "bbox": bbox,
                "bbox_xywh": [x, y, w, h],
                "img_w": img_w,
                "img_h": img_h,
                "gt_segmentation_mask": binary_mask_orig_res
            })
        return samples

    def _crop_roi_with_padding(self, img, bbox, padding_ratio=0.1, min_size=16):
        """
        Crop ROI region based on bbox and add proportional boundary expansion

        Args:
            img: Original image (numpy array, H x W x C or H x W)
            bbox: [x1, y1, x2, y2] format的边界框
            padding_ratio: Boundary expansion ratio (0.1 表示每边扩展10%)
            min_size: ROI minimum size

        Returns:
            roi_img: Cropped ROI image
            new_bbox: bbox of cropped ROI in its own image (relative coordinates)
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox


        bbox_w = x2 - x1
        bbox_h = y2 - y1


        pad_w = int(bbox_w * padding_ratio)
        pad_h = int(bbox_h * padding_ratio)


        crop_x1 = max(0, x1 - pad_w)
        crop_y1 = max(0, y1 - pad_h)
        crop_x2 = min(w, x2 + pad_w)
        crop_y2 = min(h, y2 + pad_h)


        roi_img = img[crop_y1:crop_y2, crop_x1:crop_x2]


        roi_h, roi_w = roi_img.shape[:2]
        if roi_h < min_size or roi_w < min_size:

            scale = max(min_size / roi_h, min_size / roi_w)
            new_pad_w = int(bbox_w * scale * padding_ratio)
            new_pad_h = int(bbox_h * scale * padding_ratio)
            crop_x1 = max(0, x1 - new_pad_w)
            crop_y1 = max(0, y1 - new_pad_h)
            crop_x2 = min(w, x2 + new_pad_w)
            crop_y2 = min(h, y2 + new_pad_h)
            roi_img = img[crop_y1:crop_y2, crop_x1:crop_x2]


        new_x1 = x1 - crop_x1
        new_y1 = y1 - crop_y1
        new_x2 = x2 - crop_x1
        new_y2 = y2 - crop_y1

        roi_h, roi_w = roi_img.shape[:2]
        new_bbox = [
            max(0, new_x1),
            max(0, new_y1),
            min(roi_w, new_x2),
            min(roi_h, new_y2)
        ]

        return roi_img, new_bbox, (crop_x1, crop_y1, crop_x2, crop_y2)

    def _crop_segmentation_mask(self, mask, crop_coords):

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
        return mask[crop_y1:crop_y2, crop_x1:crop_x2]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        img_id = sample["image_id"]

        try:
            if self.is_grayscale:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None: raise FileNotFoundError(f"Unable to read grayscale image: {img_path}")
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None: raise FileNotFoundError(f"Unable to read color images: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"Failed to load image {img_path} (ID: {img_id}): {e}. Return placeholder.")
            num_channels = 1 if self.is_grayscale else 3
            return {
                "image": torch.zeros((num_channels, 32, 128), dtype=torch.float32),
                "text_seq": torch.full((self.max_len,), self.char2idx["<PAD>"], dtype=torch.long),
                "label": "", "bbox": torch.IntTensor([0, 0, 0, 0]), "image_id": img_id,
                "area": torch.tensor([0.0], dtype=torch.float32)
            }


        bbox_transformed = sample["bbox"]
        gt_segmentation_mask = sample.get("gt_segmentation_mask")

        if self.use_roi_crop:

            img_for_aug, bbox_transformed, crop_coords = self._crop_roi_with_padding(
                img,
                bbox_transformed,
                padding_ratio=self.roi_crop_padding,
                min_size=self.roi_min_size
            )


            if gt_segmentation_mask is not None and gt_segmentation_mask.size > 0:
                gt_segmentation_mask = self._crop_segmentation_mask(gt_segmentation_mask, crop_coords)
        else:

            img_for_aug = img.copy()


        if self.transform:
            try:

                img_transformed, bbox_transformed = self.transform(img_for_aug, bbox_transformed)

                if img_transformed.size == 0 or img_transformed.shape[0] <= 0 or img_transformed.shape[1] <= 0:
                    print(f"Warning: image {img_id} Empty after augmentation，Use image before augmentation。")
                else:
                    img_for_aug = img_transformed

            except Exception as e:
                print(f"Data augmentation失败 {img_path} (ID: {img_id}): {e}. Using Original image。")

        final_img = img_for_aug
        final_h, final_w = final_img.shape[:2]

        final_bbox_for_item = bbox_transformed

        if final_img.size == 0 or final_h <= 0 or final_w <= 0:
            print(f"Error: Empty after ROI processing {img_path} (ID: {img_id}). Return placeholder.")
            num_channels = 1 if self.is_grayscale else 3
            return {
                "image": torch.zeros((num_channels, 32, 128), dtype=torch.float32),
                "text_seq": torch.full((self.max_len,), self.char2idx["<PAD>"], dtype=torch.long),
                "label": "", "bbox": torch.IntTensor([0, 0, 0, 0]), "image_id": img_id,
                "area": torch.tensor([0.0], dtype=torch.float32)
            }


        text_label = sample["label"]
        text_seq = [self.char2idx["<SOS>"]]
        for char_val in text_label:

            text_seq.append(self.char2idx.get(char_val, self.char2idx["<PAD>"]))
        text_seq.append(self.char2idx["<EOS>"])

        padded_text_seq = torch.full((self.max_len,), self.char2idx["<PAD>"], dtype=torch.long)
        seq_len_to_copy = min(len(text_seq), self.max_len)
        padded_text_seq[:seq_len_to_copy] = torch.tensor(text_seq[:seq_len_to_copy], dtype=torch.long)


        if not isinstance(final_img, np.ndarray):
            final_img = np.array(final_img)
        if self.is_grayscale:
            if final_img.ndim == 3 and final_img.shape[2] == 3:
                final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
            if final_img.ndim == 2:
                roi_tensor = torch.from_numpy(final_img).float().unsqueeze(0)
            else:
                raise ValueError(f"Expecting grayscale image [H,W] or [1,H,W]，but got {final_img.shape}")
        else:
            if final_img.ndim == 2:
                final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
            if final_img.ndim == 3 and final_img.shape[2] == 3:
                roi_tensor = torch.from_numpy(final_img.transpose((2, 0, 1))).float()
            else:
                raise ValueError(f"Expecting color image [H,W,C]，but got {final_img.shape}")

        roi_tensor = roi_tensor / 255.0  # Normalize to [0, 1]
        return {
            "image": roi_tensor,
            "text_seq": padded_text_seq,
            "label": text_label,
            "bbox": final_bbox_for_item,
            "image_id": img_id,
            "area": sample.get("area", torch.tensor([0.0], dtype=torch.float32)),
            "gt_segmentation_mask": gt_segmentation_mask
        }

    def __len__(self):
        return len(self.samples)

    def get_group_ids(self):
        """

        """
        if not self.grouping_config or 'cluster_centers' not in self.grouping_config:

            return torch.zeros(len(self.samples), dtype=torch.int64)

        print("Assigning group IDs to samples using adaptive clustering centers...")

        # Recover scaler and centers from loaded config
        scaler_mean = np.array(self.grouping_config['scaler_mean'])
        scaler_scale = np.array(self.grouping_config['scaler_scale'])
        centers_scaled = np.array(self.grouping_config['cluster_centers'])

        group_ids = []
        for sample in self.samples:
            try:
                img_info = self.coco.loadImgs(sample['image_id'])[0]
                width, height = img_info['width'], img_info['height']

                # ★ If using ROI cropping, should group based on ROI dimensions
                if self.use_roi_crop:
                    bbox = sample['bbox']
                    # Calculate ROI size with padding
                    bbox_w = bbox[2] - bbox[0]
                    bbox_h = bbox[3] - bbox[1]
                    pad_w = int(bbox_w * self.roi_crop_padding)
                    pad_h = int(bbox_h * self.roi_crop_padding)
                    width = bbox_w + 2 * pad_w
                    height = bbox_h + 2 * pad_h

                if height > 0 and width > 0:
                    feature_vector = np.array([[width / height, np.log(width * height)]])
                    # Use same standardization as during analysis
                    scaled_feature = (feature_vector - scaler_mean) / scaler_scale

                    distances = np.linalg.norm(scaled_feature - centers_scaled, axis=1)
                    group_id = np.argmin(distances)
                    group_ids.append(group_id)
                else:
                    group_ids.append(-1)  # Abnormal dimensions

            except Exception as e:
                print(f"Warning: Unable to load sample {sample.get('image_id', 'N/A')} image information, will assign to default group -1. Error: {e}")
                group_ids.append(-1)

        return torch.tensor(group_ids, dtype=torch.int64)

    def _initialize_grouping_config(self, ann_path):
        """
        """
        # Create unique cache filename based on annotation filename
        roi_suffix = "_roi" if self.use_roi_crop else ""
        cache_filename = Path(ann_path).stem + f"_grouping_cache{roi_suffix}.pkl"
        cache_path = Path(self.model_cfg.get('grouping_cache_dir', './configs/cache/')).joinpath(cache_filename)

        if cache_path.exists():
            print(f"Loading Grouping configuration: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.grouping_config = pickle.load(f)
            return

        print(f"Grouping cache not found, performing one-time analysis for {ann_path} ...")
        if not _SKLEARN_AVAILABLE:
            print("Error: scikit-learn not installed, cannot perform automatic grouping analysis. Please run 'pip install scikit-learn'")
            self.model_cfg['use_aspect_ratio_grouping'] = False
            return

        features = []
        for sample in self.samples:
            img_info = self.coco.loadImgs(sample['image_id'])[0]
            width, height = img_info['width'], img_info['height']


            if self.use_roi_crop:
                bbox = sample['bbox']
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                pad_w = int(bbox_w * self.roi_crop_padding)
                pad_h = int(bbox_h * self.roi_crop_padding)
                width = bbox_w + 2 * pad_w
                height = bbox_h + 2 * pad_h

            if height > 0 and width > 0:
                features.append([width / height, np.log(width * height)])

        if not features:
            self.model_cfg['use_aspect_ratio_grouping'] = False
            return

        features = np.array(features)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        num_groups = self.model_cfg.get('num_groups', 8)


        kmeans: KMeans = KMeans(n_clusters=num_groups, random_state=42, n_init='auto').fit(scaled_features)

        self.grouping_config = {
            'type': 'kmeans_adaptive',
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }


        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self.grouping_config, f)

    def type_transfer(self, ann):
        cV2 = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
               'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
               '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u',
               'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

        source_chars = []

        for index in range(self.max_len):
            if (ann['rec'][index] != 96):
                source_chars.append(cV2[ann['rec'][index]])
            else:
                break
        ann['rec'] = ann['rec'][:len(source_chars)]
        ann['rec'] = source_chars
        return ann
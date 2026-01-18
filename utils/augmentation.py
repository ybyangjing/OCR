import random
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T


class MeterAugment:
    def __init__(self,is_resize = False,use_small_dataset_augmentation = False):
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.2,
            hue=0.1
        )

        self.is_resize = is_resize

    def __call__(self, img_np, bbox_abs):

        original_dtype = img_np.dtype
        is_grayscale = (img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1))


        if is_grayscale and img_np.ndim == 3:
            img_np = np.squeeze(img_np, axis=-1)


        if random.random() > 0.3:
            if is_grayscale:
                pil_img = Image.fromarray(img_np, mode='L')
            else:
                pil_img = Image.fromarray(img_np, mode='RGB')

            pil_img = self.color_jitter(pil_img)
            img_np = np.array(pil_img)

            if is_grayscale and img_np.ndim == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        h, w = img_np.shape[:2]
        current_bbox = list(bbox_abs)



        if random.random() > 0.5:
            kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])
            sigma = random.uniform(0.5, 2.0)  # Random sigma
            img_np = cv2.GaussianBlur(img_np, kernel_size, sigma)


        if random.random() > 0.5:
            img_np = self._simulate_glare(img_np, is_grayscale)


        img_np = np.clip(img_np, 0, 255).astype(original_dtype)


        if current_bbox[2] <= current_bbox[0] or current_bbox[3] <= current_bbox[1]:
            print(f"Warning: bbox invalid after augmentation: {current_bbox}, will reset to full image.")
            final_h_clip, final_w_clip = img_np.shape[:2]
            current_bbox = [0, 0, final_w_clip, final_h_clip]

        return img_np, current_bbox

    def _simulate_glare(self, img_np, is_grayscale):
        h, w = img_np.shape[:2]
        if h == 0 or w == 0: return img_np

        glare_mask_single_channel = np.zeros((h, w), dtype=np.float32)

        cx = random.randint(0, w - 1) if w > 0 else 0
        cy = random.randint(0, h - 1) if h > 0 else 0

        # Ensure radius calculation is safe
        min_dim_for_radius = min(h, w)
        if min_dim_for_radius <= 0: return img_np


        lower_radius_bound = min(10, max(1, min_dim_for_radius // 4))
        upper_radius_bound = max(lower_radius_bound + 1, min_dim_for_radius // 2)

        radius = random.randint(lower_radius_bound, upper_radius_bound)
        cv2.circle(glare_mask_single_channel, (cx, cy), radius, 1.0, -1)

        glare_intensity = random.uniform(0.3, 0.8)  # Adjust intensity
        glare_effect = glare_mask_single_channel * glare_intensity

        img_float = img_np.astype(np.float32)
        if not is_grayscale:
            glare_effect = np.expand_dims(glare_effect, axis=-1)

        glared_img = img_float * (1.0 - glare_effect) + 255.0 * glare_effect
        return np.clip(glared_img, 0, 255).astype(img_np.dtype)

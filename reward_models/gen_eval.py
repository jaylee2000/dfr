"""
Local geneval evaluation using Mask2Former (object detection) and OpenCLIP (color classification).

Adapted from DiffusionNFT/flow_grpo/gen_eval.py.
Requires: mmdet (>=3.0), mmcv (>=2.0), open-clip-torch, clip-benchmark
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image, ImageOps

COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
]

GENEVAL_TAGS = [
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr",
]

THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_OBJECT_NAMES_PATH = os.path.join(_REPO_ROOT, "prompts", "geneval", "object_names.txt")
_MASK2FORMER_CFG = "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco"
_MASK2FORMER_CKPT = "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth"


# ── helpers ──────────────────────────────────────────────────────────────────


def _box_area(box):
    return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)


def _compute_iou(box_a, box_b):
    inter = _box_area(
        [
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3]),
        ]
    )
    union = _box_area(box_a) + _box_area(box_b) - inter
    return inter / union if union else 0


def _open_clip_cache_kwargs():
    """Return an explicit OpenCLIP cache override without changing its default."""
    cache_dir = os.environ.get("RAM_OPEN_CLIP_CACHE_DIR")
    return {"cache_dir": cache_dir} if cache_dir else {}


def _relative_position(obj_a, obj_b):
    """Position of A relative to B, accounting for object dimensions."""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    rels = set()
    if dx < -0.5:
        rels.add("left of")
    if dx > 0.5:
        rels.add("right of")
    if dy < -0.5:
        rels.add("above")
    if dy > 0.5:
        rels.add("below")
    return rels


class _ImageCrops(torch.utils.data.Dataset):
    """Crops detected objects from an image for color classification."""

    def __init__(self, image, objects, transform):
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color="#999")
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        image = image.crop(box[:4])
        return self._transform(image), 0


# ── evaluator ────────────────────────────────────────────────────────────────


class GenevalEvaluator:
    """Evaluates images against geneval metadata using local Mask2Former + OpenCLIP."""

    CKPT_DIR = "models"

    def __init__(self, device="cpu"):
        import mmdet
        from mmdet.apis import init_detector

        config_path = os.path.join(
            os.path.dirname(mmdet.__file__),
            f".mim/configs/mask2former/{_MASK2FORMER_CFG}.py",
        )
        ckpt_path = os.path.join(self.CKPT_DIR, _MASK2FORMER_CKPT)
        self.detector = init_detector(config_path, ckpt_path, device=str(device))

        import open_clip
        from clip_benchmark.metrics import zeroshot_classification as zsc

        zsc.tqdm = lambda it, *a, **kw: it
        self._zsc = zsc

        self.clip_model, _, self._clip_transform = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            device=str(device),
            **_open_clip_cache_kwargs(),
        )
        self._tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.device = device

        with open(_OBJECT_NAMES_PATH) as f:
            self.classnames = [line.strip() for line in f]

        self._color_classifiers: dict = {}

    def to(self, device):
        device = str(device)
        if device != str(self.device):
            self.device = device
            self.detector = self.detector.to(device)
            self.clip_model = self.clip_model.to(device)
            self._color_classifiers.clear()
        return self

    # ── public ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(self, pil_images, metadatas, only_strict=True):
        """
        Evaluate a batch of PIL images against their geneval metadata.

        Returns:
            only_strict=True  → list[float]  (continuous scores for training)
            only_strict=False → dict[str, list[float]]  with per-tag breakdown
        """
        from mmdet.apis import inference_detector

        results = inference_detector(self.detector, [np.array(img) for img in pil_images])
        if not isinstance(results, list):
            results = [results]

        scores, strict_list, accuracy_list, tags = [], [], [], []
        for result, pil_img, meta in zip(results, pil_images, metadatas):
            detected = self._postprocess_detections(result, meta)
            image = ImageOps.exif_transpose(pil_img)

            is_strict, score, _ = self._evaluate_strict(image, detected, meta)
            strict_list.append(1.0 if is_strict else 0.0)
            scores.append(score)
            tags.append(meta["tag"])

            if not only_strict:
                is_correct, _ = self._evaluate_lenient(image, detected, meta)
                accuracy_list.append(1.0 if is_correct else 0.0)

        if only_strict:
            return scores

        out = {
            "score": scores,
            "strict": strict_list,
            "accuracy": accuracy_list,
        }
        for tag in GENEVAL_TAGS:
            out[f"{tag}_strict"] = [s if t == tag else float("nan") for s, t in zip(strict_list, tags)]
            out[f"{tag}"] = [a if t == tag else float("nan") for a, t in zip(accuracy_list, tags)]
        return out

    # ── detection post-processing (mmdet 3.x) ────────────────────────────

    def _postprocess_detections(self, result, metadata):
        """Convert mmdet 3.x DetDataSample into per-class object lists."""
        pred = result.pred_instances
        bboxes = pred.bboxes.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        masks = pred.masks.cpu().numpy() if hasattr(pred, "masks") else None

        threshold = COUNTING_THRESHOLD if metadata["tag"] == "counting" else THRESHOLD

        detected = {}
        for idx, classname in enumerate(self.classnames):
            cls_mask = labels == idx
            cls_scores = scores[cls_mask]
            cls_bboxes = bboxes[cls_mask]
            cls_masks = masks[cls_mask] if masks is not None else None

            ordering = np.argsort(cls_scores)[::-1]
            ordering = ordering[cls_scores[ordering] > threshold]
            ordering = ordering[:MAX_OBJECTS].tolist()

            objects = []
            while ordering:
                best = ordering.pop(0)
                box_with_score = np.append(cls_bboxes[best], cls_scores[best])
                mask = cls_masks[best] if cls_masks is not None else None
                objects.append((box_with_score, mask))
                ordering = [
                    o
                    for o in ordering
                    if NMS_THRESHOLD == 1 or _compute_iou(cls_bboxes[best], cls_bboxes[o]) < NMS_THRESHOLD
                ]
            if objects:
                detected[classname] = objects
        return detected

    # ── color classification ─────────────────────────────────────────────

    def _color_classify(self, image, bboxes, classname):
        if classname not in self._color_classifiers:
            self._color_classifiers[classname] = self._zsc.zero_shot_classifier(
                self.clip_model,
                self._tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    "a photo of a {c} object",
                ],
                str(self.device),
            )
        clf = self._color_classifiers[classname]
        dataset = _ImageCrops(image, bboxes, self._clip_transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=4,
        )
        with torch.no_grad():
            pred, _ = self._zsc.run_classification(
                self.clip_model,
                clf,
                loader,
                str(self.device),
            )
            return [COLORS[i.item()] for i in pred.argmax(1)]

    # ── strict evaluation (training) ─────────────────────────────────────

    def _evaluate_strict(self, image, objects, metadata):
        """Exact count matching, continuous reward in [0, 1]."""
        correct = True
        reason = []
        rewards = []
        matched_groups = []

        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found)) / req["count"])

            if len(found) != req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}=={req['count']}, found {len(found)}")
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = self._color_classify(image, found, classname)
                    rewards.append(1 - abs(req["count"] - colors.count(req["color"])) / req["count"])
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}"
                            f"=={req['count']}, "
                            f"found {colors.count(req['color'])}"
                        )

                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        pos_ok = True
                        for obj in found:
                            for target_obj in matched_groups[target_group]:
                                rels = _relative_position(obj, target_obj)
                                if expected_rel not in rels:
                                    correct = matched = pos_ok = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} "
                                        f"target, found "
                                        f"{' and '.join(rels) or 'overlap'}"
                                    )
                                    break
                            if not matched:
                                break
                        rewards.append(1.0 if pos_ok else 0.0)

            matched_groups.append(found if matched else None)

        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, " f"found {len(objects[classname])}")

        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, "\n".join(reason)

    # ── lenient evaluation (eval) ────────────────────────────────────────

    def _evaluate_lenient(self, image, objects, metadata):
        """>= count matching, binary pass/fail."""
        correct = True
        reason = []
        matched_groups = []

        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found = objects.get(classname, [])[: req["count"]]

            if len(found) < req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, " f"found {len(found)}")
            else:
                if "color" in req:
                    colors = self._color_classify(image, found, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}"
                            f">={req['count']}, "
                            f"found {colors.count(req['color'])}"
                        )

                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found:
                            for target_obj in matched_groups[target_group]:
                                rels = _relative_position(obj, target_obj)
                                if expected_rel not in rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} "
                                        f"target, found "
                                        f"{' and '.join(rels) or 'overlap'}"
                                    )
                                    break
                            if not matched:
                                break

            matched_groups.append(found if matched else None)

        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, " f"found {len(objects[classname])}")

        return correct, "\n".join(reason)

from typing import Dict, Tuple
from collections import OrderedDict
import torch
import detectron2.utils.comm as comm

def intersection(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (ground_truth.bool() & prediction.bool()).float()


def union(ground_truth: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (ground_truth.bool() | prediction.bool()).float()


def compute_iou(ground_truth: torch.Tensor, prediction: torch.Tensor) -> float:
    num_intersection = float(torch.sum(intersection(ground_truth, prediction)))
    num_union = float(torch.sum(union(ground_truth, prediction)))
    iou = 0.0 if num_union == 0 else num_intersection / num_union
    return iou


class PQStatCategory:
    def __init__(self, is_thing=True):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.n = 0
        self.is_thing = is_thing

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        self.n += pq_stat_cat.n
        return self

    @property
    def as_metric(self):
        return {"iou": self.iou, "tp": self.tp, "fp": self.fp, "fn": self.fn, "n": self.n}

    def __repr__(self):
        return self.as_metric

class PanopticReconstructionQuality:
    def __init__(self, metadata, matching_threshold=0.25, ignore_labels=None, reduction="mean"):
        super().__init__()

        # Ignore freespace label and ceiling
        if ignore_labels is None:
            ignore_labels = [0, 12]
        self.ignore_labels = ignore_labels

        self.matching_threshold = matching_threshold
        # self.extract_mesh = extract_mesh

        self.categories = {}

        for item in metadata.class_info:
            self.categories[item["trainId"]] = PQStatCategory(item["isthing"] == 1)
        
        self.categories[0] = PQStatCategory(True)

        self.reduction = reduction
        self.metadata = metadata
        self.class_id_to_name = OrderedDict()
        self.class_id_to_name[0] = "void"
        for item in self.metadata.class_info:
            self.class_id_to_name[item["trainId"]] = item["name"]

    def add(self, prediction: Dict[int, Tuple[torch.Tensor, int]], ground_truth: Dict[int, Tuple[torch.Tensor, int]]) -> Dict:
        matched_ids_ground_truth = set()
        matched_ids_prediction = set()

        per_sample_result = {}
        for item in self.metadata.class_info:
            per_sample_result[item["trainId"]] = PQStatCategory(item["isthing"] == 1)

        # True Positives
        for ground_truth_instance_id, (ground_truth_instance_mask, ground_truth_semantic_label) in ground_truth.items():
            self.categories[ground_truth_semantic_label].n += 1
            per_sample_result[ground_truth_semantic_label].n += 1

            for prediction_instance_id, (prediction_instance_mask, prediction_semantic_label) in prediction.items():

                # 0: Check if prediction was already matched
                if prediction_instance_id in matched_ids_prediction:
                    continue

                # 1: Check if they have the same label
                are_same_category = ground_truth_semantic_label == prediction_semantic_label

                if not are_same_category:
                    # self.logger.info(f"{ground_truth_instance_id} vs {prediction_instance_id} --> Not same category {ground_truth_semantic_label} vs {prediction_semantic_label}")
                    continue

                # 2: Compute overlap and check if they are overlapping enough
                overlap = compute_iou(ground_truth_instance_mask, prediction_instance_mask)
                is_match = overlap > self.matching_threshold

                if is_match:
                    self.categories[ground_truth_semantic_label].iou += overlap
                    self.categories[ground_truth_semantic_label].tp += 1

                    per_sample_result[ground_truth_semantic_label].iou += overlap
                    per_sample_result[ground_truth_semantic_label].tp += 1

                    matched_ids_ground_truth.add(ground_truth_instance_id)
                    matched_ids_prediction.add(prediction_instance_id)
                    break

        # False Negatives
        for ground_truth_instance_id, (_, ground_truth_semantic_label) in ground_truth.items():
            # 0: Check if ground truth has not yet been matched
            if ground_truth_instance_id not in matched_ids_ground_truth:
                # self.logger.info(f"Not matched, counted as FN: {ground_truth_instance_id}, num voxels: {mask.sum()}")
                self.categories[ground_truth_semantic_label].fn += 1
                per_sample_result[ground_truth_semantic_label].fn += 1

        # False Positives
        for prediction_instance_id, (_, prediction_semantic_label) in prediction.items():
            # 0: Check if prediction has not yet been matched
            if prediction_instance_id not in matched_ids_prediction:
                # self.logger.info(f"Not matched, counted as FP: {prediction_instance_id}, num voxels: {mask.sum()}")
                self.categories[prediction_semantic_label].fp += 1
                per_sample_result[prediction_semantic_label].fp += 1

        return per_sample_result

    def add_sample(self, sample):
        for k in sample.keys():
            if k in self.categories:
                self.categories[k] += sample[k]

    def reduce(self) -> Dict:        
        if comm.get_world_size() > 1:
            # Distributed mode, need to gather data
            categories = comm.gather(self.categories, dst=0)
            if not comm.is_main_process():
                return None
            for cat in categories[1:]:
                for class_label in self.categories.keys():
                    self.categories[class_label] += cat[class_label]

        if self.reduction == "mean":
            return self._reduce_mean()

        return None

    def _reduce_mean(self):
        pq, sq, rq, n = 0, 0, 0, 0

        per_class_results = {}

        for class_label, class_stats in self.categories.items():
            iou = class_stats.iou
            tp = class_stats.tp
            fp = class_stats.fp
            fn = class_stats.fn
            num_samples = class_stats.n

            if tp + fp + fn == 0:
                per_class_results[class_label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'n': 0}
                continue

            if num_samples > 0:
                n += 1
                pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
                sq_class = iou / tp if tp != 0 else 0
                rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
                per_class_results[class_label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'n': num_samples}
                pq += pq_class
                sq += sq_class
                rq += rq_class

        results = OrderedDict()
        results.update({'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n})

        for label, per_class_result in per_class_results.items():
            if per_class_result["n"] > 0:
                label = self.class_id_to_name[label]
                results[label] = {
                    "pq": per_class_result["pq"],
                    "sq": per_class_result["sq"],
                    "rq": per_class_result["rq"],
                    "n": per_class_result["n"],
                }

        return results

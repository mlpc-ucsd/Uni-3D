import logging
import torch

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .metrics import PanopticReconstructionQuality


class Front3DEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection, segmentation and meshes
    outputs.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self.metric = PanopticReconstructionQuality(self._metadata)
        

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            # Add to metric
            self.metric.add(output["instance_info_pred"], input["instance_info_gt"])

    def evaluate(self):
        ret = self.metric.reduce()

        if not comm.is_main_process():
            return

        classes = self.metric.class_id_to_name.values()

        print(
            "Result:\n"
            "PQ     SQ      RQ      N\n"
            f"{ret['pq']*100:.2f}  {ret['sq']*100:.2f}   {ret['rq']*100:.2f}   {ret['n']}"
        )
        max_len = max([len(cls) for cls in classes])
        for cls in classes:
            if not cls in ret.keys():
                continue
            print(cls + " " * (max_len+2-len(cls)) + "PQ     SQ      RQ      N")
            print(" " * (max_len+2) + f"{ret[cls]['pq']*100:.2f}  {ret[cls]['sq']*100:.2f}   {ret[cls]['rq']*100:.2f}   {ret[cls]['n']}")

        return ret


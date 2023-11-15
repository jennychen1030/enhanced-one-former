from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, create_backbone, create_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import semantic_seg_postprocessing
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import handle_cuda_oom

from .modeling.loss import SegmentationCriterion
from .modeling.assignment  import OptimalMatcher
from einops import rearrange
from .modeling.transformer_decoder.textual_transformer import TextTransformer
from .modeling.transformer_decoder.oneformer_decoder_mlp import MultiLayerPerceptron
from oneformer.data.simple_tokenizer import BasicTokenizer, BasicTokenize

@META_ARCH_REGISTRY.register()
class NewOneFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(
        self,
        *,
        backbone_network: Backbone,
        segmentation_head: nn.Module,
        task_specific_mlp: nn.Module,
        language_encoder: nn.Module,
        language_projection: nn.Module,
        segmentation_criterion: nn.Module,
        context_embedding: nn.Embedding,
        query_count: int,
        mask_threshold: float,
        segment_overlap_threshold: float,
        dataset_metadata,
        divisibility_factor: int,
        preprocess_before_inference: bool,
        mean_pixel_value: Tuple[float],
        std_pixel_value: Tuple[float],
        # inference
        enable_semantic: bool,
        enable_panoptic: bool,
        enable_instance: bool,
        enable_detection: bool,
        topk_predictions_per_image: int,
        text_sequence_length: int,
        max_text_length: int,
        demo_mode: bool,
    ):
        """
        Constructor for NewOneFormer.
        
        Args:
            backbone_network: a backbone_network module, must follow detectron2's backbone_network interface
            segmentation_head: a module that predicts semantic segmentation from backbone_network features
            segmentation_criterion: a module that defines the loss
            query_count: int, number of queries
            mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            segment_overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            dataset_metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            divisibility_factor: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            preprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            mean_pixel_value, std_pixel_value: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            enable_semantic: bool, whether to output semantic segmentation prediction
            enable_instance: bool, whether to output instance segmentation prediction
            enable_panoptic: bool, whether to output panoptic segmentation prediction
            topk_predictions_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone_network = backbone_network
        self.segmentation_head = segmentation_head
        self.task_specific_mlp = task_specific_mlp
        self.language_encoder = language_encoder
        self.language_projection = language_projection
        self.context_embedding = context_embedding
        self.segmentation_criterion = segmentation_criterion
        self.query_count = query_count
        self.segment_overlap_threshold = segment_overlap_threshold
        self.mask_threshold = mask_threshold
        self.dataset_metadata = dataset_metadata
        if divisibility_factor < 0:
            divisibility_factor = self.backbone_network.size_divisibility
        self.divisibility_factor = divisibility_factor
        self.preprocess_before_inference = preprocess_before_inference
        self.register_buffer("mean_pixel_value", torch.Tensor(mean_pixel_value).view(-1, 1, 1), False)
        self.register_buffer("std_pixel_value", torch.Tensor(std_pixel_value).view(-1, 1, 1), False)

        # additional args
        self.enable_semantic = enable_semantic
        self.enable_instance = enable_instance
        self.enable_panoptic = enable_panoptic
        self.enable_detection = enable_detection
        self.topk_predictions_per_image = topk_predictions_per_image

        self.text_tokenizer = BasicTokenize(BasicTokenizer(), max_text_length=max_text_length)
        self.task_tokenizer = BasicTokenize(BasicTokenizer(), max_text_length=text_sequence_length)
        self.demo_mode = demo_mode

        self.thing_indices = [k for k in self.dataset_metadata.thing_dataset_id_to_contiguous_id.keys()]

        if not self.enable_semantic:
            assert self.preprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        # Creating components from configuration
        backbone_network = create_backbone(cfg)
        segmentation_head = create_sem_seg_head(cfg, backbone_network.output_shape())

        # Setup for training
        if cfg.MODEL.IS_TRAIN:
            language_encoder = TextTransformer(
                               context_length=cfg.MODEL.TEXT_ENCODER.CONTEXT_LENGTH,
                               width=cfg.MODEL.TEXT_ENCODER.WIDTH,
                               layers=cfg.MODEL.TEXT_ENCODER.NUM_LAYERS,
                               vocab_size=cfg.MODEL.TEXT_ENCODER.VOCAB_SIZE)
            
            language_projection = MultiLayerPerceptron(
                                  language_encoder.width, 
                                  cfg.MODEL.ONE_FORMER.HIDDEN_DIM, 
                                  cfg.MODEL.ONE_FORMER.HIDDEN_DIM, 
                                  cfg.MODEL.TEXT_ENCODER.PROJ_NUM_LAYERS)
            if cfg.MODEL.TEXT_ENCODER.N_CTX > 0:
                context_embedding = nn.Embedding(cfg.MODEL.TEXT_ENCODER.N_CTX, cfg.MODEL.TEXT_ENCODER.WIDTH)
            else:
                context_embedding = None
        else:
            language_encoder = None
            language_projection = None
            context_embedding = None

        task_specific_mlp = MultiLayerPerceptron(
                            cfg.INPUT.TASK_SEQ_LEN, 
                            cfg.MODEL.ONE_FORMER.HIDDEN_DIM,
                            cfg.MODEL.ONE_FORMER.HIDDEN_DIM, 2)

        # Loss parameters:
        deep_supervision = cfg.MODEL.ONE_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.ONE_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.ONE_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.ONE_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.ONE_FORMER.MASK_WEIGHT
        contrastive_weight = cfg.MODEL.ONE_FORMER.CONTRASTIVE_WEIGHT
        
        # building segmentation_criterion
        matcher = OptimalMatcher(
                  cost_class=class_weight,
                  cost_mask=mask_weight,
                  cost_dice=dice_weight,
                  num_points=cfg.MODEL.ONE_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, 
                        "loss_dice": dice_weight, "loss_contrastive": contrastive_weight}

        
        if deep_supervision:
            dec_layers = cfg.MODEL.ONE_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "contrastive"]

        segmentation_criterion = SegmentationCriterion(
            segmentation_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            contrast_temperature=cfg.MODEL.ONE_FORMER.CONTRASTIVE_TEMPERATURE,
            losses=losses,
            num_points=cfg.MODEL.ONE_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.ONE_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.ONE_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone_network": backbone_network,
            "segmentation_head": segmentation_head,
            "task_specific_mlp": task_specific_mlp,
            "context_embedding": context_embedding,
            "language_encoder": language_encoder,
            "language_projection": language_projection,
            "segmentation_criterion": segmentation_criterion,
            "query_count": cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES,
            "mask_threshold": cfg.MODEL.TEST.OBJECT_MASK_THRESHOLD,
            "segment_overlap_threshold": cfg.MODEL.TEST.OVERLAP_THRESHOLD,
            "dataset_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "divisibility_factor": cfg.MODEL.ONE_FORMER.SIZE_DIVISIBILITY,
            "preprocess_before_inference": (
                cfg.MODEL.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.TEST.PANOPTIC_ON
                or cfg.MODEL.TEST.INSTANCE_ON
            ),
            "mean_pixel_value": cfg.MODEL.PIXEL_MEAN,
            "std_pixel_value": cfg.MODEL.PIXEL_STD,
            # inference
            "enable_semantic": cfg.MODEL.TEST.SEMANTIC_ON,
            "enable_instance": cfg.MODEL.TEST.INSTANCE_ON,
            "enable_panoptic": cfg.MODEL.TEST.PANOPTIC_ON,
            "enable_detection": cfg.MODEL.TEST.DETECTION_ON,
            "topk_predictions_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "text_sequence_length": cfg.INPUT.TASK_SEQ_LEN,
            "max_text_length": cfg.INPUT.MAX_SEQ_LEN,
            "demo_mode": cfg.MODEL.IS_DEMO,
        }

    @property
    def device(self):
        """
        Property to get the device where the model tensors are allocated.
        """
        return self.mean_pixel_value.device
    def semantic_inference(self, mask_cls, mask_pred):
        """
        Perform semantic segmentation inference.
        Args:
            mask_cls: Classification scores for each mask.
            mask_pred: Predicted masks.
        Returns:
            Tensor: Semantic segmentation results.
        """

        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        """
        Perform panoptic segmentation inference.
        Args:
            mask_cls: Classification scores for each mask.
            mask_pred: Predicted masks.
        Returns:
            Tuple: Panoptic segmentation results and segment info.
        """

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.segmentation_head.num_classes) & (scores > self.mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.dataset_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.segment_overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, task_type):
        """
        Perform instance segmentation inference.
        Args:
            mask_cls: Classification scores for each mask.
            mask_pred: Predicted masks.
            task_type: Type of task for the input.
        Returns:
            Instances: Instance segmentation results.
        """

        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.segmentation_head.num_classes, device=self.device).unsqueeze(0).repeat(self.query_count, 1).flatten(0, 1)
        
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.query_count, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.topk_predictions_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.segmentation_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # Only consider scores with confidence over [self.mask_threshold] for demo
        if self.demo_mode:
            keep = scores_per_image > self.mask_threshold
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.enable_panoptic:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.dataset_metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        
        if 'ade20k' in self.dataset_metadata.name and not self.demo_mode and "instance" in task_type:
            for i in range(labels_per_image.shape[0]):
                labels_per_image[i] = self.thing_indices.index(labels_per_image[i].item())

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        if self.enable_detection:
            # Uncomment the following to get boxes from masks (this is slow)
            result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
    
    def encode_text(self, text):
        """
        Method to encode textual inputs.
        Args:
            text (Tensor): Text input tensor.
        Returns:
            dict: A dictionary with text encoding results.
        """
        assert text.ndim in [2, 3], text.ndim
        batch_size = text.shape[0]
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)

        # Encoding text
        encoded_features = self.language_encoder(text)
        text_features = self.language_projection(encoded_features)
        if text.ndim == 3:
            text_features = rearrange(text_features, '(b n) c -> b n c', n=num_text)
            if self.context_embedding is not None:
                context_features = self.context_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
                text_features = torch.cat([text_features, context_features], dim=1)
        
        return {"texts": text_features}
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.mean_pixel_value) / self.std_pixel_value for x in images]
        images = ImageList.from_tensors(images, self.divisibility_factor)

        tasks = torch.cat([self.task_tokenizer(x["task"]).to(self.device).unsqueeze(0) for x in batched_inputs], dim=0)
        tasks = self.task_specific_mlp(tasks.float())

        features = self.backbone_network(images.tensor)
        outputs = self.segmentation_head(features, tasks)

        if self.training:
            texts = torch.cat([self.text_tokenizer(x["text"]).to(self.device).unsqueeze(0) for x in batched_inputs], dim=0)
            texts_x = self.encode_text(texts)
            outputs = {**outputs, **texts_x}

            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            losses = self.segmentation_criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.segmentation_criterion.weight_dict:
                    losses[k] *= self.segmentation_criterion.weight_dict[k]
                else:
                    losses.pop(k)
            return losses
        else:
            # Handle inference here
            mask_class_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # Upsample masks to match image sizes
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            del outputs

            processed_results = []
            for (mask_cls, mask_pred, input_per_image, image_size) in enumerate(
                    zip(mask_class_results, mask_pred_results, batched_inputs, images.image_sizes)):
                
                height, width = input_per_image.get("height", image_size[0]), input_per_image.get("width", image_size[1])
                processed_result = {}

                if self.preprocess_before_inference:
                    mask_pred = handle_cuda_oom(semantic_seg_postprocessing)(
                        mask_pred, image_size, height, width)
                    mask_cls = mask_cls.to(mask_pred.device)

                # semantic segmentation inference
                if self.enable_semantic:
                    semantic_result = handle_cuda_oom(self.semantic_inference)(mask_cls, mask_pred)
                    if not self.preprocess_before_inference:
                        semantic_result = handle_cuda_oom(semantic_seg_postprocessing)(semantic_result, image_size, height, width)
                    processed_result["sem_seg"] = semantic_result

                # panoptic segmentation inference
                if self.enable_panoptic:
                    panoptic_result = handle_cuda_oom(self.panoptic_inference)(mask_cls, mask_pred)
                    processed_result["panoptic_seg"] = panoptic_result
                
                # instance segmentation inference
                if self.enable_instance:
                    instance_result = handle_cuda_oom(self.instance_inference)(mask_cls, mask_pred, input_per_image["task"])
                    processed_result["instances"] = instance_result

                processed_results.append(processed_result)

            return processed_results
    
    def prepare_targets(self, targets, images):
        """
        Prepare ground truth targets for training.
        Args:
            targets: Ground truth instances.
            images: Batched input images.
        Returns:
            list[dict]: Prepared targets.
        """
        
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets
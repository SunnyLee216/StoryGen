# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download
import os
import supervision as sv
from PIL import Image, ImageDraw, ImageFont

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from controlnet_aux.pidi import PidiNetDetector


class GroundingAndSegmentation:
    def __init__(self, grounding_model, sam_model, pidinet_model,device= 'cpu'):
        self.grounding_model = grounding_model
        self.sam_model = sam_model
        self.pidinet_model = pidinet_model
        self.device = device
    @staticmethod
    def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        # Usage
        # ckpt_repo_id = "ShilongLiu/GroundingDINO"
        # ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        # ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        # groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
 
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file) 
        args.device = device
        model = build_model(args)
        
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model


    def detect_and_segment(self, image, text_prompt, box_threshold = 0.3, text_threshold = 0.25):
        # Run Grounding-DINO for detection
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        # Run SAM for segmentation
        # Convert boxes to the format required by SAM
        # Prepare SAM model
        self.sam_model.set_image(image)
        H, W, _ = image.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        # Transform boxes and run SAM for segmentation
        transformed_boxes = self.sam_model.transform.apply_boxes_torch(boxes_xyxy.to(self.device), image.shape[:2])
        masks, _, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )
        image_mask = masks[0][0].cpu().numpy()
        image_mask_pil = Image.fromarray(image_mask)

        edges = self.pidinet_model(
            image_mask_pil, detect_resolution=1024, image_resolution=1024, apply_filter=True
        )
        # Process and return the results
        return boxes, masks.cpu(), edges
class ImageComposer:
    @staticmethod
    def composing_dense_condition(layouts, edges, output_size=(512, 1024)):
        # Create a blank image with the specified output size
        composed_image = Image.new('RGB', output_size, (255, 255, 255))

        for layout, edge in zip(layouts, edges):
            object_name, (x, y, width, height) = layout

            # Resize the edge image to match the layout size
            resized_edge = edge.resize((width, height))

            # Paste the resized edge into the composed image at the specified location
            composed_image.paste(resized_edge, (x, y))

        return composed_image
    



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
    groundingdino_model = GroundingAndSegmentation.load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    
    
    
    sam_checkpoint = 'sam_vit_h_4b8939.pth'
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    pidinet_model = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

    grounding_and_segmentation = GroundingAndSegmentation(groundingdino_model, sam_predictor, pidinet_model,device='cuda')
    image = ...  # Load or provide an image
    text_prompt = "Your Text Prompt"
    boxes, masks = grounding_and_segmentation.detect_and_segment(image, text_prompt)
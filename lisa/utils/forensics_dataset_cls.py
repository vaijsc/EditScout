import glob
import json
import os
import random
import sys
sys.path.append(".")
sys.path.append("..")
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from PIL import Image
import random
from ..model.llava import conversation as conversation_lib
from ..model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,EXPLANATORY_FORENSIC_QUESTION_LIST,
                    SHORT_QUESTION_LIST, FORENSIC_QUESTION_LIST, 
                    FORENSICS_QUESTIONS, FORENSICS_QUESTIONS_EXPLANATORY, FORENSIC_ANSWER, FORENSIC_ANSWER_EXPLANATORY
                    
                    )

#   TODO 1: Fix the code for the new format questions/answers
#   TODO 2: Write the pipeline for using auth image
#   TODO 3: Test the pipeline with explanatory

class ForensicDatasetCLS(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        annotation_path,
        tokenizer,
        vision_tower,
        samples_per_epoch,
        precision: str = "fp32",
        image_size: int = 224,
        exclude_val=False,
        explanatory=False,
        is_val=False,
        sam_only = True,
        use_authentic=False,
        use_system_prompt=False
    ):
        self.is_val = is_val
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory

        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = FORENSICS_QUESTIONS
        self.answer_list = FORENSIC_ANSWER
        self.sam_only = sam_only
        self.anno_list = json.load(open(annotation_path, "r"))[:]

        self.edit_list, self.auth_list = [], []
        for item in self.anno_list:
            self.edit_list.append(item["edit_path"])
            self.auth_list.append(item["auth_path"])

        print("number of forensics samples: ", len(self.anno_list))

        if explanatory:
            self.explanatory_answer = FORENSIC_ANSWER_EXPLANATORY
            self.explanatory_question_list = FORENSICS_QUESTIONS_EXPLANATORY
            self.img_to_explanation = {}
            for item in self.anno_list:
                img_name = item["edit_path"]
                self.img_to_explanation[img_name] = {
                    "query": random.choice(self.explanatory_question_list),
                    "outputs": item["instruction"],
                }

            print("len(self.img_to_explanation): ", len(self.img_to_explanation))

        self.images_list = self.auth_list + self.edit_list

    def __len__(self):
        return len(self.images_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        
        #   Edit list: 0 -> len(edit_list) - 1
        #   Auth list: len(edit_list) -> end

        image_path = self.images_list[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        #   Create class for image path: 0 - Authentic, 1 - Edited
        #   Edit class
        if idx >= len(self.auth_list):
            CLS = torch.ones(1,).float()
        else:
            #   Auth class
            CLS = torch.zeros(1,).float()

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")
        image_clip = image_clip["pixel_values"][0]

        sampled_sents = [""]
        mask_path = self.anno_list[idx % len(self.edit_list)]["mask_path"]
        sampled_masks = np.array(Image.open(mask_path)).astype("float32")
        
        if sampled_masks.max() == 255.0:
            sampled_masks = sampled_masks / 255.0

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image_name = image_path

        choice = 0 
        if self.explanatory and image_name in self.img_to_explanation.keys():
            choice = 1
        
        questions = []
        answers = []

        question_template = random.choice(self.short_question_list)
        questions.append(question_template)

        #   QUESTION and ASNWER section
        # add explanation if applicable
        img_name = image_path

        if self.explanatory and img_name in self.img_to_explanation.keys():
            if choice == 0:  # [SEG] token
                answer_template = self.answer_list[0]
                if isinstance(answer_template, str):
                    # answers.append(random.choice(self.answer_list))       #   old source of lisa
                    answers.append(answer_template)
                elif isinstance(answer_template, dict):
                    if self.is_val:
                        answers.append(" ")
                    else:
                        answers.append(answer_template["yes"])

            elif choice == 1:  # [SEG] token + text answer
                image_name = image_path
                answer = self.img_to_explanation[image_name]["outputs"]

                explanatory_template = self.explanatory_answer[0]
                if isinstance(explanatory_template, str):
                    answer = explanatory_template.format(answer=answer)
                elif isinstance(explanatory_template, dict):
                    answer = explanatory_template["yes"].format(answer=answer)

                questions[-1] = (
                    DEFAULT_IMAGE_TOKEN
                    + "\n"
                    + " {}".format(random.choice(self.explanatory_question_list))
                )
                answers.append(answer)
            else:
                raise ValueError("Not implemented yet.")
        else:
            answer_template = self.answer_list[0]
            if isinstance(answer_template, str):
                # answers.append(random.choice(self.answer_list))       #   old source of lisa
                answers.append(answer_template)
            elif isinstance(answer_template, dict):
                answers.append(answer_template["yes"]) 

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        if len(masks.shape) == 2:
            masks.unsqueeze_(0)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        if self.is_val:
            inference=True
        else:
            inference=False

        
        # print(CLS, image_path)
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
            False,
            inference,
            CLS
        )
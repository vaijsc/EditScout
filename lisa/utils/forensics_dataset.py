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

from ..model.llava import conversation as conversation_lib
from ..model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,EXPLANATORY_FORENSIC_QUESTION_LIST,
                    SHORT_QUESTION_LIST, FORENSIC_QUESTION_LIST, 
                    FORENSICS_QUESTIONS, FORENSICS_QUESTIONS_EXPLANATORY, FORENSIC_ANSWER, FORENSIC_ANSWER_EXPLANATORY,
                    SYSTEM_PROMPT, FORENSICS_QUESTIONS_RANDOM, FORENSICS_QUESTIONS_EXPLANATORY_RANDOM
                    )

class ForensicDataset(torch.utils.data.Dataset):
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
        use_system_prompt=False,
        random_question=False,
    ):
        self.is_val = is_val
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.use_system_prompt = use_system_prompt
        self.use_authentic = use_authentic 
        self.random_question = random_question

        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        
        self.short_question_list = FORENSICS_QUESTIONS
        if self.random_question:
            self.short_question_list = FORENSICS_QUESTIONS_RANDOM

        self.answer_list = FORENSIC_ANSWER
        self.sam_only = sam_only
        self.anno_list = json.load(open(annotation_path, "r"))
        
        print("number of forensics samples: ", len(self.anno_list))
        
        if explanatory:
            self.explanatory_answer = FORENSIC_ANSWER_EXPLANATORY
            self.explanatory_question_list = FORENSICS_QUESTIONS_EXPLANATORY
            if self.random_question:
                self.explanatory_question_list = FORENSICS_QUESTIONS_EXPLANATORY_RANDOM
            self.img_to_explanation = {}
            for item in self.anno_list:
                img_name = item["edit_path"]
                self.img_to_explanation[img_name] = {
                    "query": random.choice(self.explanatory_question_list),
                    "outputs": item["instruction"],
                }
            print("The number of image explanation", len(self.img_to_explanation))

    def __len__(self):
        return len(self.anno_list)

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
        # images, jsons = self.reason_seg_data
        # idx = random.randint(0, len(images) - 1)
        # image_path = images[idx]
        # json_path = jsons[idx]

        annotation = self.anno_list[idx]

        image_path = annotation["edit_path"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # preprocess image for clip

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")
        image_clip = image_clip["pixel_values"][0]

        sampled_sents = [""]
        mask_path = annotation["mask_path"]
        sampled_masks = np.array(Image.open(mask_path)).astype("float32")
        
        if sampled_masks.max() == 255.0:
            sampled_masks = sampled_masks / 255.0
        
        if sampled_masks.max() > 1.0:
            raise ValueError("Something wrong in rescaling the mask")

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image_name = image_path

        choice = 0 
        if self.explanatory and image_name in self.img_to_explanation.keys():
            choice = 1
        
        questions = []
        answers = []


        question_template = random.choice(self.short_question_list)

        if self.use_system_prompt:
            question_template = SYSTEM_PROMPT + question_template
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
                    answers.append(answer_template["yes"])

            elif choice == 1:  # [SEG] token + text answer
                image_name = image_path
                answer = self.img_to_explanation[image_name]["outputs"]

                explanatory_template = self.explanatory_answer[0]
                if isinstance(explanatory_template, str):
                    answer = explanatory_template.format(answer=answer)
                elif isinstance(explanatory_template, dict):
                    answer = explanatory_template["yes"].format(answer=answer.lower())

                if self.use_system_prompt:
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + "{} {}".format(SYSTEM_PROMPT, random.choice(self.explanatory_question_list))
                    )
                else:
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

            if self.use_system_prompt:
                if SYSTEM_PROMPT not in conversations[i]:
                    raise ValueError("System prompt not in the conversation")
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
        
        if self.is_val == True and self.use_authentic:
            raise ValueError("Not using authentic in validation")

        print(conversations)

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
            -1
        )

# class ForensicDataset(torch.utils.data.Dataset):
#     pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
#     pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
#     img_size = 1024
#     ignore_label = 255

#     def __init__(
#         self,
#         annotation_path,
#         tokenizer,
#         vision_tower,
#         samples_per_epoch,
#         precision: str = "fp32",
#         image_size: int = 224,
#         exclude_val=False,
#         explanatory=False,
#         is_val=False,
#         sam_only=True,
#         use_authentic=False
#     ):
#         self.is_val = is_val
#         self.exclude_val = exclude_val
#         self.samples_per_epoch = samples_per_epoch
#         self.explanatory = explanatory
#         self.use_authentic = use_authentic

#         if self.is_val:
#             self.use_authentic = False

#         print("Validation set: {} and use authentic: {}".format(self.is_val, self.use_authentic))
#         self.image_size = image_size
#         self.tokenizer = tokenizer
#         self.precision = precision
#         self.transform = ResizeLongestSide(image_size)
#         self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

#         self.short_question_list = FORENSICS_QUESTIONS
#         self.answer_list = FORENSIC_ANSWER
#         self.sam_only = sam_only
#         self.anno_list = json.load(open(annotation_path, "r"))
#         if self.use_authentic:
#             self.tamper_list = [item["edit_path"] for item in self.anno_list]        
#             assert len(self.tamper_list) == len(self.anno_list)

#         print("number of forensics samples: ", len(self.anno_list))
        
#         if explanatory:
#             self.explanatory_answer = FORENSIC_ANSWER_EXPLANATORY
#             self.explanatory_question_list = FORENSICS_QUESTIONS_EXPLANATORY
#             self.img_to_explanation = {}
#             for item in self.anno_list:
#                 img_name = item["edit_path"]
#                 self.img_to_explanation[img_name] = {
#                     "query": random.choice(self.explanatory_question_list),
#                     "outputs": item["instruction"],
#                 }
#             print("The number of image explanation", len(self.img_to_explanation))

#     def __len__(self):
#         if self.use_authentic:
#             return len(self.anno_list) * 2
#         return len(self.anno_list)

#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         """Normalize pixel values and pad to a square input."""
#         # Normalize colors
#         x = (x - self.pixel_mean) / self.pixel_std

#         # Pad
#         h, w = x.shape[-2:]
#         padh = self.img_size - h
#         padw = self.img_size - w
#         x = F.pad(x, (0, padw, 0, padh))
#         return x

#     def __getitem__(self, idx):
#         # images, jsons = self.reason_seg_data
#         # idx = random.randint(0, len(images) - 1)
#         # image_path = images[idx]
#         # json_path = jsons[idx]

#         input_authentic = False
#         if self.use_authentic and idx >= len(self.tamper_list):
#             input_authentic = True
#             idx -= len(self.tamper_list) 

#         annotation = self.anno_list[idx]
#         #   Input the edit image
#         if not input_authentic:
#             image_path = annotation["edit_path"]

#         #   Input the authentic image
#         else:
#             image_path = annotation["auth_path"]

#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         ori_size = image.shape[:2]

#         # preprocess image for clip
#         image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")
#         image_clip = image_clip["pixel_values"][0]

#         #   Create binary mask for edit image if `input_authentic = True`
#         #   else create the zeros mask for authentic image if `input_authentic = False`
#         sampled_sents = [""]
#         mask_path = annotation["mask_path"]
#         sampled_masks = np.array(Image.open(mask_path)).astype("float32") 
        
#         if sampled_masks.max() == 255.0:
#             sampled_masks = sampled_masks / 255.0

#         masks = np.stack(sampled_masks, axis=0)
#         masks = torch.from_numpy(masks)
#         if len(masks.shape) == 2:
#             masks.unsqueeze_(0)
#         label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
#         image = self.transform.apply_image(image)  # preprocess image for sam
#         image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
#         resize = image.shape[:2]

#         image_name = image_path

#         choice = 0 
#         if self.explanatory and image_name in self.img_to_explanation.keys():
#             choice = 1
        
#         questions = []
#         answers = []

#         question_template = random.choice(self.short_question_list)
#         questions.append(question_template)

        
#         #   QUESTION and ASNWER section
#         #   add explanation if applicable
#         img_name = image_path
#         if self.explanatory and img_name in self.img_to_explanation.keys():
#             if choice == 0:  # [SEG] token
#                 answer_template = self.answer_list[0]
#                 if isinstance(answer_template, str):
#                     # answers.append(random.choice(self.answer_list))       #   old source of lisa
#                     answers.append(answer_template)
#                 elif isinstance(answer_template, dict):
#                     if not input_authentic:
#                         answers.append(answer_template["yes"])
#                     else:
#                         answers.append(answer_template["no"])


#             elif choice == 1:  # [SEG] token + text answer
#                 image_name = image_path
#                 answer = self.img_to_explanation[image_name]["outputs"]

#                 explanatory_template = self.explanatory_answer[0]
#                 if isinstance(explanatory_template, str):
#                     answer = explanatory_template.format(answer=answer)
#                 elif isinstance(explanatory_template, dict):
#                     if not input_authentic:         #   Not using authentic -> edit
#                         answer = explanatory_template["yes"].format(answer=answer)
#                     else:
#                         answer = explanatory_template["no"]     #   Authentic image


#                 questions[-1] = (
#                     DEFAULT_IMAGE_TOKEN
#                     + "\n"
#                     + " {}".format(random.choice(self.explanatory_question_list))
#                 )
#                 answers.append(answer)
#             else:
#                 raise ValueError("Not implemented yet.")
#         else:
#             answer_template = self.answer_list[0]
#             if isinstance(answer_template, str):
#                 # answers.append(random.choice(self.answer_list))       #   old source of lisa
#                 answers.append(answer_template)
#             elif isinstance(answer_template, dict):
#                 if not input_authentic:
#                     answers.append(answer_template["yes"])
#                 else:
#                     answers.append(answer_template["no"])


#         conversations = []
#         conv = conversation_lib.default_conversation.copy()
#         roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
#         i = 0
#         while i < len(questions):
#             conv.messages = []
#             conv.append_message(conv.roles[0], questions[i])
#             conv.append_message(conv.roles[1], answers[i])
#             conversations.append(conv.get_prompt())
#             i += 1


#         if self.is_val:
#             inference=True
#         else:
#             inference=False
        
#         if self.use_authentic == False and input_authentic == True:
#             raise ValueError("Not input authentic because we don't use authentic images.")

#         return (
#             image_path,
#             image,
#             image_clip,
#             conversations,
#             masks,
#             label,
#             resize,
#             questions,
#             sampled_sents,
#             input_authentic,
#             inference,
#         )

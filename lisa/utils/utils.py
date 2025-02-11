from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

TEMPLATE_ID=0

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

FORENSIC_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the edited region in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the manipulated region in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is the edited region in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is edited region in this image? Please output segmentation mask.",
]

EXPLANATORY_FORENSIC_QUESTION_LIST = [
    "Please output segmentation mask and can you guess and describe the edit process.",
    "Please output segmentation mask. Please explain how this image is edited.",
    "Please output segmentation mask and describe the process of editing image.",
    "Please output segmentation mask and the instruction to produce the edited image.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

SYSTEM_PROMPT = "Inpainted or edited images often show distinct signs, such as noise pattern inconsistencies, where inpainted areas have different noise levels or distributions. Edge inconsistencies are also common, with added or removed objects displaying halos or mismatched sharpness. From this provided information, please answer the following question. "



TEMPLATES = [
    "Is this image edited. If yes, can you segment the edited region",
    "Is there any indication that this image has been altered? If yes, could you segment the modified regions",
    "Can you determine if this image has been manipulated? If so, please highlight the altered areas",
    "Please analyze this image for any signs of editing. If the image has been edited, identify and segment the edited portions",
]

TEMPLATES_EXPLANATORY = [
    "Is this image edited. If yes, can you segment the edited region and give the instruction used to edit this image.",
    "Is there any indication that this image has been altered? If yes, could you segment the modified regions and provide a detailed explanation of the editing process?",
    "Can you determine if this image has been manipulated? If so, please highlight the altered areas and describe the techniques used to modify the image.",  
    "Please analyze this image for any signs of editing. If the image has been edited, identify and segment the edited portions, and outline the steps taken to achieve the edits.",
]

"""
Done:   [
    

]
"""


print("Template: {}".format(TEMPLATES[TEMPLATE_ID]))

#   Option 1:
FORENSICS_QUESTIONS = [
    DEFAULT_IMAGE_TOKEN + "\n" + TEMPLATES[TEMPLATE_ID],
]

#   Option 2:
FORENSICS_QUESTIONS_EXPLANATORY = [
    TEMPLATES_EXPLANATORY[TEMPLATE_ID]
]

FORENSICS_QUESTIONS_RANDOM = [
    DEFAULT_IMAGE_TOKEN + "\n" +     "Is this image edited. If yes, can you segment the edited region",
    DEFAULT_IMAGE_TOKEN + "\n" +     "Is there any indication that this image has been altered? If yes, could you segment the modified regions",
    DEFAULT_IMAGE_TOKEN + "\n" +     "Can you determine if this image has been manipulated? If so, please highlight the altered areas and describe the techniques used to modify the image.",  
    DEFAULT_IMAGE_TOKEN + "\n" +     "Please analyze this image for any signs of editing. If the image has been edited, identify and segment the edited portions",

]

#   Option 2:
FORENSICS_QUESTIONS_EXPLANATORY_RANDOM = [
    "Is this image edited. If yes, can you segment the edited region and give the instruction used to edit this image.",
    "Is there any indication that this image has been altered? If yes, could you segment the modified regions and provide a detailed explanation of the editing process?",
    "Can you determine if this image has been manipulated? If so, please highlight the altered areas and describe the techniques used to modify the image.",  
    "Please analyze this image for any signs of editing. If the image has been edited, identify and segment the edited portions, and outline the steps taken to achieve the edits.",
]

print("Template for forensic explanatory: {}".format(FORENSICS_QUESTIONS_EXPLANATORY))

#   Option 1:
FORENSIC_ANSWER = [
    {
        "yes": "Yes, the segmentation result is [SEG].",
        "no": "No. This image is authentic.",
    },
]

# FORENSIC_ANSWER = [
#     "Yes, the segmentation result is [SEG].",
# ]

#   Option 2
FORENSIC_ANSWER_EXPLANATORY = [
    {
        "yes": "Yes, the segmentation result is [SEG]. The instruction used to edit this image could be {answer}",
        "no": "No. This image is authentic.",
    },
    
]
# FORENSIC_ANSWER_EXPLANATORY = [
#     "Yes, the segmentation result is [SEG]. The instruction used to edit this image could be {answer}", 
# ]





class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict

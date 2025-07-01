# import math
# import warnings
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# def bbox_multi_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, WIoU=False, UIoU=False, EfficiCIoU=False, XIoU=False, bat=0, is_Focaler='None', eps=1e-7):
#     """
#     Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

#     Args:
#         box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
#         box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
#         xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
#                                (x1, y1, x2, y2) format. Defaults to True.
#         GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
#         DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
#         CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
#         eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
#         # @from MangoAI &3836712GKcH2717GhcK.

#     Returns:
#         (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
#     """

#     Inner = False
#     if Inner == False:
#         if xywh:  # transform from xywh to xyxy
#             (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
#             w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#             b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#             b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#         else:  # x1, y1, x2, y2 = box1
#             b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
#             b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
#             w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
#             w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

#         epoch = bat
#         if UIoU:
#             # print('UIoU')
#             print(epoch)
#             # define the center point for scaling
#             bb1_xc = x1
#             bb1_yc = y1
#             bb2_xc = x2
#             bb2_yc = y2
#             # attenuation mode of hyperparameter "ratio"
#             linear = True
#             cosine = False
#             fraction = False 
#             # assuming that the total training epochs are 300, the "ratio" changes from 2 to 0.5
#             if linear:
#                 ratio = -0.005 * epoch + 2
#             elif cosine:
#                 ratio = 0.75 * math.cos(math.pi * epoch / 300) + 1.25
#             elif fraction:
#                 ratio = 200 / (epoch + 100)
#             else:
#                 ratio = 0.5
#             ww1, hh1, ww2, hh2 = w1 * ratio, h1 * ratio, w2 * ratio, h2 * ratio
#             bb1_x1, bb1_x2, bb1_y1, bb1_y2 = bb1_xc - (ww1 / 2), bb1_xc + (ww1 / 2), bb1_yc - (hh1 / 2), bb1_yc + (hh1 / 2)
#             bb2_x1, bb2_x2, bb2_y1, bb2_y2 = bb2_xc - (ww2 / 2), bb2_xc + (ww2 / 2), bb2_yc - (hh2 / 2), bb2_yc + (hh2 / 2)
#             # assign the value back to facilitate subsequent calls
#             w1, h1, w2, h2 = ww1, hh1, ww2, hh2
#             b1_x1, b1_x2, b1_y1, b1_y2 = bb1_x1, bb1_x2, bb1_y1, bb1_y2
#             b2_x1, b2_x2, b2_y1, b2_y2 = bb2_x1, bb2_x2, bb2_y1, bb2_y2
#             # CIoU = True
#     # ---------------------------------------------------------------------------------------------------------------


#         # Intersection area
#         inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
#             b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
#         ).clamp_(0)

#         # Union Area
#         union = w1 * h1 + w2 * h2 - inter + eps

#         # IoU
#         iou = inter / union
#     #  --------------原始IoU----------------------
#     else:
#         (x1, y1, w1, h1) = box1.chunk(4, -1)
#         (x2, y2, w2, h2) = box2.chunk(4, -1)
#         # @from MangoAI &3836712GKcH2717GhcK.
#         ratio = 0.8
#         w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#         b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#         b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#         inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_*ratio, x1 + w1_*ratio,\
#                                                                 y1 - h1_*ratio, y1 + h1_*ratio
#         inner_b2_x1,inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_*ratio, x2 + w2_*ratio,\
#                                                                 y2 - h2_*ratio, y2 + h2_*ratio
#         inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
#                     (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
#         union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - inner_inter + eps
#         iou = inner_inter/union
#     #  --------------Inner----------------------


#     # FocalerIoU 改进
#     Focaler = False
#     if Focaler:
#         d=0.0
#         u=0.95
#         print('use Focaler系列🍈')
#         iou = ((iou - d) / (u - d)).clamp(0, 1)

#     if CIoU or DIoU or GIoU or EIoU or SIoU or EfficiCIoU or XIoU:
#         cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
#         ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
#         if CIoU or DIoU or EIoU or SIoU or EfficiCIoU or XIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             c2 = cw**2 + ch**2 + eps  # convex diagonal squared
#             rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
#             if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + eps))
#                 print('CIoU🚀')
#                 return iou - (rho2 / c2 + v * alpha)  # CIoU🚀
#             elif SIoU:
#                 s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
#                 s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
#                 sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
#                 sin_alpha_1 = torch.abs(s_cw) / sigma
#                 sin_alpha_2 = torch.abs(s_ch) / sigma
#                 threshold = pow(2, 0.5) / 2
#                 sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
#                 angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
#                 rho_x = (s_cw / cw) ** 2
#                 rho_y = (s_ch / ch) ** 2
#                 gamma = angle_cost - 2
#                 distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
#                 omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
#                 omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
#                 shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
#                 print('SIoU🚀')
#                 return iou - 0.5 * (distance_cost + shape_cost)# SIoU🚀
#             elif EIoU:
#                 v = torch.pow(1 / (1 + torch.exp(-(w2 / h2))) - 1 / (1 + torch.exp(-(w1 / h1))), 2)
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + eps))
#                 print('EIoU🚀')
#                 return iou - (rho2 / c2 + v * alpha)# EIoU🚀
#             elif EfficiCIoU:
#                 # @from MangoAI &3836712GKcH2717GhcK.
#                 c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
#                 rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
#                         (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
#                 w_dis=torch.pow(b1_x2-b1_x1-b2_x2+b2_x1, 2)
#                 h_dis=torch.pow(b1_y2-b1_y1-b2_y2+b2_y1, 2)
#                 cw2=torch.pow(cw , 2)+eps
#                 ch2=torch.pow(ch , 2)+eps
#                 v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + eps))
#                 print('EfficiCIoU🚀')
#                 return iou - (rho2 / c2 + w_dis/cw2+h_dis/ch2 + v * alpha)
#             elif XIoU:# @from MangoAI &3836712GKcH2717GhcK.
#                 c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
#                 rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
#                 beta = 1
#                 q2 = (1 + torch.exp(-(w2 / h2)))
#                 q1 = (1 + torch.exp(-(w1 / h1)))
#                 v = torch.pow(1 / q2 - 1 / q1, 2)
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + eps)) * beta
#                 print('XIoU🚀')
#                 return iou - (rho2 / c2 + v * alpha)
#             print('DIoU🚀')
#             return iou - rho2 / c2  # DIoU🚀
#         c_area = cw * ch + eps  # convex area
#         print('GIoU🚀')
#         return iou - (c_area - union) / c_area  # GIoU🚀 https://arxiv.org/pdf/1902.09630.pdf
#     return iou  # 🚀IoU



import math
import torch

def bbox_multi_iou_spatiu(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, EIoU=False, 
                         SIoU=False, EfficiCIoU=False, XIoU=False, 
                         SPTIoU=False, eps=1e-7):
    """
    Calculate various IoU metrics, including the custom SPTIoU for spikelet detection.

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to False.
        GIoU, DIoU, CIoU, EIoU, SIoU, WIoU, UIoU, EfficiCIoU, XIoU (bool, optional): Flags to calculate respective IoU variants.
        SPTIoU (bool, optional): If True, calculate the custom SPTIoU. Defaults to False.
        bat (int, optional): Current batch index or epoch, used for certain IoU variants like UIoU.
        is_Focaler (str, optional): Placeholder for Focaler-related functionality.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: IoU, GIoU, DIoU, CIoU, EIoU, SIoU, EfficiCIoU, XIoU, SPTIoU values depending on the specified flags.
    """

    # Convert boxes from xywh to xyxy if necessary
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # Initialize the loss with standard IoU
    loss = iou

    # EIoU and SPTIoU Calculation
    if EIoU or SPTIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # Convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # Convex height
        c2 = cw ** 2 + ch ** 2 + eps  # Convex diagonal squared

        # Center points
        center_b1_x = (b1_x1 + b1_x2) / 2
        center_b1_y = (b1_y1 + b1_y2) / 2
        center_b2_x = (b2_x1 + b2_x2) / 2
        center_b2_y = (b2_y1 + b2_y2) / 2

        # Center distance using Manhattan distance
        center_distance_x = torch.abs(center_b2_x - center_b1_x)
        center_distance_y = torch.abs(center_b2_y - center_b1_y)
        center_distance = center_distance_x + center_distance_y
        rho2 = center_distance / (cw + ch + eps)  # Normalized Manhattan distance

        # Width and height differences
        w_dis = torch.pow(w1 - w2, 2)
        h_dis = torch.pow(h1 - h2, 2)

        # Aspect ratio term
        v = torch.pow(1 / (1 + torch.exp(-(w2 / h2))) - 1 / (1 + torch.exp(-(w1 / h1))), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))

        # Original EIoU
        eiou = iou - (rho2 / c2 + w_dis / cw ** 2 + h_dis / ch ** 2 + alpha * v)

        # If SPTIoU is enabled, add direction alignment penalty
        if SPTIoU:
            # 1. Compute center vector components
            v_x = center_b2_x - center_b1_x
            v_y = center_b2_y - center_b1_y

            # 2. Compute the norm of the center vector
            v_norm = torch.sqrt(v_x ** 2 + v_y ** 2 + eps)

            # 3. Compute cosine similarity with vertical direction (0,1)
            cos_similarity = v_y / v_norm  # Cosine similarity with vertical

            # 4. Clamp to ensure numerical stability
            direction_alignment = cos_similarity.clamp(-1, 1)  # Range [-1, 1]

            # 5. Convert to penalty term
            angle_penalty = 1 - direction_alignment  # Range [0, 2]

            # 6. Scale the penalty
            lambda_angle = 0.5  # Hyperparameter, adjust as needed
            angle_penalty = lambda_angle * angle_penalty  # Range [0, lambda_angle * 2]

            # 7. Combine EIoU with angle penalty to get SPTIoU
            sptiou = eiou - angle_penalty

            # Update loss
            loss = sptiou
            print("SPTIoU")
        else:
            # Update loss with EIoU
            loss = eiou
            print("EIoU")
    # Other IoU variants (GIoU, DIoU, CIoU, SIoU, EfficiCIoU, XIoU)
    if not (EIoU or SPTIoU):  # Only compute other IoUs if EIoU and SPTIoU are not enabled
        if CIoU or DIoU or GIoU or SIoU or EfficiCIoU or XIoU:
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # Convex width
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # Convex height
            c2 = cw**2 + ch**2 + eps  # Convex diagonal squared

            # Compute center distance squared for DIoU, CIoU, etc.
            rho2 = ((center_b2_x - center_b1_x) ** 2 + (center_b2_y - center_b1_y) ** 2) / 4  # Center distance squared

            if CIoU:
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                print('CIoU')
                loss = iou - (rho2 / c2 + v * alpha)  # CIoU
            elif SIoU:
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                print('SIoU')
                loss = iou - 0.5 * (distance_cost + shape_cost)  # SIoU
            elif EIoU:
                v = torch.pow(1 / (1 + torch.exp(-(w2 / h2))) - 1 / (1 + torch.exp(-(w1 / h1))), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                print('EIoU')
                loss = iou - (rho2 / c2 + v * alpha)  # EIoU
            elif EfficiCIoU:
                w_dis = torch.pow(b1_x2 - b1_x1 - b2_x2 + b2_x1, 2)
                h_dis = torch.pow(b1_y2 - b1_y1 - b2_y2 + b2_y1, 2)
                cw2 = torch.pow(cw, 2) + eps
                ch2 = torch.pow(ch, 2) + eps
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                print('EfficiCIoU')
                loss = iou - (rho2 / c2 + w_dis / cw2 + h_dis / ch2 + v * alpha)  # EfficiCIoU
            elif XIoU:
                beta = 1
                q2 = (1 + torch.exp(-(w2 / h2)))
                q1 = (1 + torch.exp(-(w1 / h1)))
                v = torch.pow(1 / q2 - 1 / q1, 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps)) * beta
                print('XIoU')
                loss = iou - (rho2 / c2 + v * alpha)  # XIoU

        # GIoU Calculation (handled if no other IoU variants are enabled)
        if not (EIoU or SPTIoU or CIoU or DIoU or SIoU or EfficiCIoU or XIoU):
            if GIoU:
                c_area = cw * ch + eps  # Convex area
                print('GIoU')
                loss = iou - (c_area - union) / c_area  # GIoU
            # Add other IoU variants like WIoU, UIoU here if needed

    # FocalerIoU and other custom IoU variants can be added similarly

    return loss  # Return the computed IoU variant loss

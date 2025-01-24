#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#



from .common_runtime import *

try:
    # Sometimes python does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def GiB(val):
    return val * 1 << 30


import cv2
import numpy as np

def fb_blur_fusion_foreground_estimator_2(image, alpha, blur_radius=90):
    """
    Estimate the foreground image by applying a blur fusion method.

    Args:
        image (numpy.ndarray): The input image.
        alpha (numpy.ndarray): The alpha matte.
        blur_radius (int, optional): The blur radius for the fusion. Default is 90.

    Returns:
        numpy.ndarray: The estimated foreground image.
    """
    alpha = alpha[:, :, None]
    foreground, blurred_background = fb_blur_fusion_foreground_estimator(
        image, image, image, alpha, blur_radius
    )
    return fb_blur_fusion_foreground_estimator(
        image, foreground, blurred_background, alpha, blur_radius=6
    )[0]


def fb_blur_fusion_foreground_estimator(image, foreground, background, alpha, blur_radius=90):
    """
    Perform blur fusion to estimate the foreground and background images.

    Args:
        image (numpy.ndarray): The input image.
        foreground (numpy.ndarray): The initial foreground estimate.
        background (numpy.ndarray): The initial background estimate.
        alpha (numpy.ndarray): The alpha matte.
        blur_radius (int, optional): The blur radius for the fusion. Default is 90.

    Returns:
        tuple: A tuple containing the estimated foreground and blurred background images.
    """
    blurred_alpha = cv2.blur(alpha, (blur_radius, blur_radius))[:, :, None]

    blurred_foreground_alpha = cv2.blur(foreground * alpha, (blur_radius, blur_radius))
    blurred_foreground = blurred_foreground_alpha / (blurred_alpha + 1e-5)

    blurred_background_alpha = cv2.blur(background * (1 - alpha), (blur_radius, blur_radius))
    blurred_background = blurred_background_alpha / ((1 - blurred_alpha) + 1e-5)

    foreground = blurred_foreground + alpha * (
            image - alpha * blurred_foreground - (1 - alpha) * blurred_background
    )
    foreground = np.clip(foreground, 0, 1)

    return foreground, blurred_background

def sigmoid(x):
    # 对 x > 0 和 x <= 0 分别进行处理，避免溢出问题
    pos_mask = x >= 0
    neg_mask = x < 0

    # 对于 x >= 0 使用稳定计算公式
    result = np.zeros_like(x)
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    # 对于 x < 0 使用等效的替代公式
    result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))

    return result

def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise ValueError(f"onnx file is not exists")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    return engine_data
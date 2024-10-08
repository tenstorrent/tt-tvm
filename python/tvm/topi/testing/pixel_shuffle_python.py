# SPDX-FileCopyrightText: © 2019-2023 The Apache Software Foundation
#
# SPDX-License-Identifier: Apache-2.0
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Pixel shuffle in python"""
import numpy as np

def pixel_shuffle_python(data, upscale_factor):
    """Pixel Shuffle operator in Python.

    Parameters
    ----------
    data : numpy.ndarray
        N-D with shape (d_0, d_1, ..., d_{N-1})

    upscale_factor : int or tuple of ints
        Integer value to be upscaled with

    Returns
    -------
    ret : numpy.ndarray
        The computed result.
    """
    b, c, h, w = data.shape
    upscale_squared = upscale_factor * upscale_factor
    assert c % upscale_squared == 0, f"The number of channels ({c}) is not divisible by upscale_factor squared ({upscale_squared})."
    oc = c // upscale_squared
    oh = h * upscale_factor
    ow = w * upscale_factor
    new_shape = (b, oc, upscale_factor, upscale_factor, h, w)
    out_shape = (b, oc, oh, ow)
    data = np.reshape(data, new_shape)
    data = np.transpose(data, (0, 1, 4, 2, 5, 3))
    out_np = np.reshape(data, out_shape)
    return out_np
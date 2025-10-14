/*!
 **************************************************************************************************
 * Deformable DETR
 * Copyright (c) 2020 SenseTime. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 **************************************************************************************************
 * Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
 **************************************************************************************************
 */

#pragma once

#include "ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "ms_deform_attn_cuda.h"
#endif

namespace groundingdino {

/**
 * Forward interface (CPU + CUDA) for the deformable attention operator.
 *
 * 参数说明（保持与原代码一致）：
 *   - value:                (N, S, nhead, C)   输入特征
 *   - spatial_shapes:       (nlevel, 2)        每个特征层的 (H, W)
 *   - level_start_index:    (nlevel)           每层在拼接后 feature map 的起始位置
 *   - sampling_loc:         (N, Lq, nhead, nlevel, npoint, 2) 采样坐标
 *   - attn_weight:          (N, Lq, nhead, nlevel, npoint)      attention 权重
 *   - im2col_step:          计算时的 batch 子块大小（默认 = min(N, 64)）
 *
 * 该函数会根据 Tensor 是否在 CUDA 上自动分派到对应实现。
 */
static inline at::Tensor
ms_deform_attn_forward(const at::Tensor &value,
                       const at::Tensor &spatial_shapes,
                       const at::Tensor &level_start_index,
                       const at::Tensor &sampling_loc,
                       const at::Tensor &attn_weight,
                       const int im2col_step) {
  // 使用新版 API 检查是否在 CUDA 上（已在 PyTorch 2.x 中废弃 .type()）
  if (value.is_cuda()) {
#ifdef WITH_CUDA
    return ms_deform_attn_cuda_forward(value,
                                      spatial_shapes,
                                      level_start_index,
                                      sampling_loc,
                                      attn_weight,
                                      im2col_step);
#else
    AT_ERROR("GroundingDINO was compiled without CUDA support");
#endif
  }

  // 这里保留原来的 “CPU not implemented” 报错，防止误用
  AT_ERROR("GroundingDINO ms_deform_attn_forward CPU implementation is not provided");
}

/**
 * Backward interface (CPU + CUDA) for the deformable attention operator.
 *
 * 与 forward 对称，返回三个梯度：
 *   - grad_value
 *   - grad_sampling_loc
 *   - grad_attn_weight
 */
static inline std::vector<at::Tensor>
ms_deform_attn_backward(const at::Tensor &value,
                        const at::Tensor &spatial_shapes,
                        const at::Tensor &level_start_index,
                        const at::Tensor &sampling_loc,
                        const at::Tensor &attn_weight,
                        const at::Tensor &grad_output,
                        const int im2col_step) {
  if (value.is_cuda()) {
#ifdef WITH_CUDA
    return ms_deform_attn_cuda_backward(value,
                                       spatial_shapes,
                                       level_start_index,
                                       sampling_loc,
                                       attn_weight,
                                       grad_output,
                                       im2col_step);
#else
    AT_ERROR("GroundingDINO was compiled without CUDA support");
#endif
  }

  AT_ERROR("GroundingDINO ms_deform_attn_backward CPU implementation is not provided");
}

} // namespace groundingdino
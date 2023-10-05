#pragma once
#include "cuda_runtime.h"

template <typename scalar_t>
void bicubic_interpolate(const scalar_t *input, scalar_t *output, int batch, int channles, int height, int width,
                         int out_height, int out_width, bool align_corners, cudaStream_t stream);
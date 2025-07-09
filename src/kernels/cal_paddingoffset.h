#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/macro.h"
#include "src/utils/tensor.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
// 其他可能需要的头文件...
void launchCalPaddingoffset(TensorWrapper<int>* padding_offset,  //   
                            TensorWrapper<int>* cum_seqlens, // 累计的句子长度
                            TensorWrapper<int>* input_lengths //actual input lens 输入长度
);
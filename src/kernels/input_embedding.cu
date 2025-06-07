#include <stdio.h>
#include "src/kernels/input_embedding.h"
#include "src/utils/cuda_debug_utils.cuh"
template<typename T>
__global__ void embeddingFunctor(const int* input_ids,
               T* output, 
               const T* embed_table,
               const int max_context_token_num,
               const int hidden_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; // 全局idx
    while (index < max_context_token_num * hidden_size) {// 线程 数据不匹配时进行循环处理 这里保证了 index.max == output.size()
        int id = input_ids[index / hidden_size]; // 多个线程处理， 将每个H内的数值分配给H个线程来做。 通过inputs 定位到目标词在词表中的id
        output[index] = embed_table[id * hidden_size + index % hidden_size]; // 多个线程并行取出
        index += blockDim.x * gridDim.x;
    }
}
template<typename T>
__global__ void embeddingFunctor_h(const int* input_ids, T* output, const T* embed_table, const int max_context_token_num, const int hidden_size) {
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tot_data = hidden_size * max_context_token_num;
    for(int idx = gidx; idx < tot_data; idx += stride) {
        int vacab_idx = input_ids[gidx / hidden_size];
        output[gidx] = embed_table[vacab_idx * hidden_size + gidx % hidden_size];
    }
}

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // INT [token num]
                          TensorWrapper<T>* output,       // FP32 [token num, hidden_size] = [token num, 4096]
                          EmbeddingWeight<T>* embed_table// FP32 [vocab_size, hidden_size]
                          ) {
    const int blockSize = 256;
    const int max_context_token_num = output->shape[0]; // token num
    const int hidden_size = output->shape[1];
    const int gridSize = 2048;
    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], "input ids 1st shape should equal to 1st shape of output");
    embeddingFunctor_h<T><<<gridSize, blockSize>>>(input_ids->data,
                                                 output->data,
                                                 embed_table->data,
                                                 max_context_token_num,
                                                 hidden_size);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}
/*显式实例化在此代码中的作用：

明确支持的精度类型：只允许 float 和 half

确保CUDA内核代码生成：为每种精度生成特定内核

避免跨编译单元问题：保证单一定义

优化编译时间：避免重复实例化

提供清晰接口：明确库支持的类型*/
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<float>* output,       
                                   EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<half>* output,       
                                   EmbeddingWeight<half>* embed_table);

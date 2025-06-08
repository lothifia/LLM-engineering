#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include "src/kernels/build_casual_mask.h"
void PrintCausalMask(const float* mask, int batch_size, int max_q_len, int max_k_len) {
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "Batch " << b << ":\n";
        for (int q = 0; q < max_q_len; ++q) {
            std::cout << "q=" << q << " | ";
            for (int k = 0; k < max_k_len; ++k) {
                // 线性 index = b * (max_q_len * max_k_len) + q * max_k_len + k
                int idx = b * max_q_len * max_k_len + q * max_k_len + k;
                std::cout << static_cast<int>(mask[idx]) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "------------------------\n";
    }
}
// (RussWong)note: this kernel's CPU implementation is absolutely right.
// when you are implementing LLMs inference on CPU, you can reuse the CPU kernel
// we compare the kernel correctnesss by eyes and result print infos
void CPUbuildCausalMask(float* mask, 
                        const int* q_lens,  //input lens, shape=[batch size]
                        const int* k_lens,  //context lens, shape=[batch size]
                        int max_q_len, 
                        int max_k_len,
                        int batch_size) {
    for(int b = 0; b < batch_size; b++){
        int start = b * max_q_len * max_k_len;
        int q = q_lens[b];
        int k = k_lens[b];
        for(int i = 0; i < max_q_len; i++) {
            for(int j = 0; j < max_k_len; j++) {
                if(j <= i + (k - q) && i < q && j < k) {
                    mask[start + i * max_k_len + j] = 1.0f;
                } else {
                    mask[start + i * max_k_len + j] = 0.0f;   
                }
            }
        }
    }
}
bool CheckResult(float* CPUres, float* GPUres, const int size) {
    for(int i = 0; i < size; i++) {
        if(fabs(CPUres[i] - GPUres[i]) > 1e-6){
            printf("the %dth res is wrong, CPU mask = %f, GPU mask = %f\n", i, CPUres[i], GPUres[i]);
            return false;
        }
    }
    return true;
}
// (RussWong)note:
// `./causalmask` to test fp32 GPU build causal mask kernel
int main() {
    const int batch_size = 1;
    const int max_q_len = 4;
    const int max_k_len = 8;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int mask_size = batch_size * max_q_len * max_k_len;
    int* h_q_lens;
    int* d_q_lens;
    h_q_lens = (int*)malloc(sizeof(int) * batch_size);
    cudaMalloc((void**)&d_q_lens, sizeof(int) * batch_size);
    int* h_k_lens;
    int* d_k_lens;
    h_k_lens = (int*)malloc(sizeof(int) * batch_size);
    cudaMalloc((void**)&d_k_lens, sizeof(int) * batch_size);

    float* d_mask;
    float* h_mask = (float*)malloc(sizeof(float) * mask_size);
    cudaMalloc((void**)&d_mask, sizeof(float) * mask_size);

    for(int i = 0; i < batch_size; i++) {
       h_q_lens[i] = 4;
    }
    for(int i = 0; i < batch_size; i++) {
       h_k_lens[i] = 8;
    }
    CHECK(cudaMemcpy(d_q_lens, h_q_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_k_lens, h_k_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    DataType type_float = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    TensorWrapper<float>* mask = new TensorWrapper<float>(Device::GPU, 
                                                        type_float,
                                                        {batch_size, max_q_len, max_k_len}, 
                                                        d_mask);
    TensorWrapper<int>* q_lens = new TensorWrapper<int>(Device::GPU, 
                                                        type_int,
                                                        {batch_size}, 
                                                        d_q_lens);
    TensorWrapper<int>* k_lens = new TensorWrapper<int>(Device::GPU, 
                                                        type_int,
                                                        {batch_size}, 
                                                        d_k_lens);
    launchBuildCausalMasks(mask, q_lens, k_lens);
    // debug info, better to retain: std::cout << "after launch kernel" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    CHECK(cudaMemcpy(h_mask, d_mask, sizeof(float) * mask_size, cudaMemcpyDeviceToHost));

    float* CPUmask = (float*)malloc(sizeof(float) * mask_size);
    CPUbuildCausalMask(CPUmask, h_q_lens, h_k_lens, max_q_len, max_k_len, batch_size);
    if (CheckResult(CPUmask, h_mask, mask_size)) {
        printf("test passed!\n");
    }
    PrintCausalMask(CPUmask, batch_size, max_q_len, max_k_len);

    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_q_lens);
    free(h_k_lens);
    free(h_mask);
    free(CPUmask);
    cudaFree(d_q_lens);
    cudaFree(d_k_lens);
    cudaFree(d_mask);
}

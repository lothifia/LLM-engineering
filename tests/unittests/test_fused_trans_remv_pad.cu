#include "src/kernels/fused_transpose_and_remv_pad.h"
#include <iostream>
// [b,h,s,d]=>[b,s,h,d]=>[num tokens,h,d]
// padding_offset.shape = [num_tokens]
// (RussWong)note: this kernel is only supporting fp32 type UT
// we compare the kernel correctnesss by eyes and result print infos
// `./test_fused_trans_remv_pad` to test fp32 kernel
int main() {
    const int batch_size = 2;
    const int head_num = 2;
    const int max_seq_len = 4;
    const int head_size = 2;
    const int num_tokens = 5;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int in_size = batch_size * head_num * max_seq_len * head_size; // 2 * 4 * 2 * 2 = 32
    const int out_size = num_tokens * head_num * head_size; // 5 * 2 * 2 = 20
    float* h_in;
    float* d_in;
    h_in = (float*)malloc(sizeof(float) * in_size);
    cudaMalloc((void**)&d_in, sizeof(float) * in_size);
    float* h_out;
    float* d_out;
    h_out = (float*)malloc(sizeof(float) * out_size);
    cudaMalloc((void**)&d_out, sizeof(float) * out_size);
    int* h_padding_offset;
    int* d_padding_offset;
    h_padding_offset = (int*)malloc(sizeof(int) * num_tokens);
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * num_tokens);

    //1st seqlen: 2, due to 1st seq, so its padding offset are all 0
    //2nd seqlen: 3, so its padding offset are all 4-2=2
    for(int i = 0; i < in_size; i++) {
       h_in[i] = i;
    }
    for(int i = 0; i < 2; i++) {
       h_padding_offset[i] = 0;
    } 
    h_padding_offset[2] = 2;  
    h_padding_offset[3] = 2;
    h_padding_offset[4] = 2;

    cudaMemcpy(d_in, h_in, sizeof(float) * in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * num_tokens, cudaMemcpyHostToDevice);

    DataType type = getTensorType<float>(); 
    DataType type_pad = getTensorType<int>(); 
    TensorWrapper<float>* in = new TensorWrapper<float>(Device::GPU, type, {batch_size, head_num, max_seq_len, head_size}, d_in);
    TensorWrapper<int>* in_pad = new TensorWrapper<int>(Device::GPU, type_pad, {num_tokens}, d_padding_offset);
    TensorWrapper<float>* out = new TensorWrapper<float>(Device::GPU, type, {num_tokens, head_num, head_size}, d_out);
    std::cout << "before launch softmax kernel" << std::endl;
    launchTransposeOutRemovePadding(in, in_pad, out);
    std::cout << "after launch softmax kernel" << std::endl;
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_out, out->data, sizeof(float) * out_size, cudaMemcpyDeviceToHost);
    // 目标逻辑布局: [b, s, h, d] = [2, 4, 2, 2]
std::cout << "--------------------------------------------------------" << std::endl;
std::cout << "Printing matrix before padding removal (logical layout [b, s, h, d]):" << std::endl;
for (int b = 0; b < batch_size; ++b) {
    std::cout << "Batch " << b << ":" << std::endl;
    for (int s = 0; s < max_seq_len; ++s) {
        printf("  Sequence %d:\n    ", s);
        for (int h = 0; h < head_num; ++h) {
            for (int d = 0; d < head_size; ++d) {
                // 计算在原始 h_in (布局为 [b, h, s, d]) 中的索引
                int original_stride_b = head_num * max_seq_len * head_size;
                int original_stride_h = max_seq_len * head_size;
                int original_stride_s = head_size;

                int index = b * original_stride_b + 
                            h * original_stride_h + 
                            s * original_stride_s + 
                            d;
                
                // 打印属于 Head h 的数据
                // 为了格式美观，这里做了判断
                if (d == 0) {
                    printf("H%d:[%5.1f, ", h, h_in[index]);
                } else {
                    printf("%5.1f] ", h_in[index]);
                }
            }
        }
        printf("\n");
    }
}
std::cout << "--------------------------------------------------------" << std::endl;

    for(int i = 0; i < num_tokens; i++) {
        printf("padding offset[%d] = %d\n", i, h_padding_offset[i]);
    }
    for(int i = 0; i < out_size; i++) {
        printf("after trans and remv pad, out[%d] = %f\n", i, h_out[i]);
    }
    
    
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_in);
    free(h_out);
    free(h_padding_offset);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_padding_offset);
}
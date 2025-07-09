#include "src/kernels/cal_paddingoffset.h"
// shape:
    //seq_lengths:[batch size]
    //cum_seqlens:[batch size + 1],first ele is 0
    //padding_offset:[batch size * max q len]
// note: the point is to calc padding offset and cum offset
// TODO: we first use serial algo, then can enhance to CUDA scan algo

__global__ void CalPaddingoffset(int*        padding_offset, 
                                int*         cum_seqlens, // 表示累积到当前坐标的的句子长度 
                                const int*   input_lengths, //actual input lens
                                const int    batch_size,
                                const int    max_q_len) {
    int ind = 0;
    int cum_offset = 0;
    int total_seqlen = 0;
    for(int b = 0; b < batch_size; b++) { // 所有句子
        int seqlen = input_lengths[b]; // 该句长度

        cum_seqlens[b] = total_seqlen; //  当前句子之前的句子长度
        // each token in one seq has same cum offset
        for (int i = 0; i < seqlen; i++) { // 对句子内的每个token
            padding_offset[ind] = cum_offset; // paddingOffest 修改成 cum_offest 表示之前累积了多少padding 即之前的prefix
            ind++; // 在padding_offest中是个长数组.
        }
        cum_offset += max_q_len - seqlen; // prefix叠加
        total_seqlen += seqlen; // 句子总长度增加
    }
    cum_seqlens[batch_size] = total_seqlen;// 跟新全部句子                            

}

__global__ void test(int* a, int* b, int length, int g) {
    printf("length: %d\n", length);
    printf("a: ");
    for(int i = 0; i < length + 1; i++) {
        printf("%d ", a[i]);
    }
    printf("\n");
    printf("b: ");
    for(int i = 0; i < g; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");
}

__global__ void ScanPadding(int* padding_offset, 
                            int* cum_seqlens, 
                            int max_q_len,
                            int batch_size) {
    // padding_offset: [batch_size, max_q_len]
    // cum_seqlens: [batch_size + 1]
    // max_q_len: max query length of all batch
    // 1 block per batch
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    int pad_offest = cum_seqlens[batch_id];
    for(int i = pad_offest + tid; i < cum_seqlens[batch_id + 1]; i += blockDim.x) {
        padding_offset[i] = max_q_len * batch_id - cum_seqlens[batch_id];
    }   
}

void launchCalPaddingoffset(TensorWrapper<int>* padding_offset, 
                            TensorWrapper<int>* cum_seqlens,
                            TensorWrapper<int>* input_lengths)//actual input lens
{
    const int batch_size = padding_offset->shape[0];      // 句子长度                      
    const int max_q_len = padding_offset->shape[1];       // 最大长度
    LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0], "input lenghts numbers should equal to padding offset bs dim!") ;                        
    LLM_CHECK_WITH_INFO(batch_size == cum_seqlens->shape[0] - 1, "cum seqlen numbers should equal to padding offset bs dim + 1!") ;                        
    // CalPaddingoffset<<<1, 1>>>( 
    //     padding_offset->data, cum_seqlens->data, input_lengths->data, batch_size, max_q_len
    // );
        // 包装原始指针
    thrust::device_ptr<int> d_input_lengths(input_lengths->data);
    thrust::device_ptr<int> d_cum_seqlens(cum_seqlens->data);

    // 前缀和计算
    thrust::exclusive_scan(d_input_lengths, d_input_lengths + batch_size + 1, d_cum_seqlens);
    const int physical_block_size = min(((max_q_len + 31) / 32) * 32, 1024);
    dim3 gird(batch_size);
    dim3 block(physical_block_size);
    ScanPadding<<<gird, block>>>(padding_offset->data, cum_seqlens->data, max_q_len, batch_size);
}
#include <cuda_runtime.h>
#include <assert.h>
#include <cstdlib>
#include "common/matrix.h"
#include "common/eval.h"
#include "common/check.h"
#include <iostream>

#define TILE_SIZE 16
#define KB *1024

__global__ void matmul(Matrix A, Matrix B, Matrix C) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    float* matA = A.elements;
    float* matB = B.elements;
    float* matC = C.elements;

    int tile_id_row = blockIdx.y;
    int tile_id_col = blockIdx.x;
    int tile_offset_row = threadIdx.y;
    int tile_offset_col = threadIdx.x;
    float sum = 0.0f;
    for(int i=0;i<gridDim.x;i++){
        tile_A[tile_offset_row][tile_offset_col] = matA[(i*TILE_SIZE+tile_offset_col)+A.width*(tile_id_row*TILE_SIZE+tile_offset_row)];
        tile_B[tile_offset_row][tile_offset_col] = matB[(tile_id_col*TILE_SIZE+tile_offset_col)+B.width*(i*TILE_SIZE+tile_offset_row)];
        __syncthreads();
        for(int j=0;j<TILE_SIZE;j++) {
            sum += tile_A[tile_offset_row][j] * tile_B[j][tile_offset_col];
        }
        __syncthreads();
    }
    matC[(tile_id_row*TILE_SIZE+tile_offset_row)*C.width + (tile_id_col*TILE_SIZE+tile_offset_col)] = sum;
}



void basic_tiled_matmul(int dim,Matrix A,Matrix B,Matrix C){
    Eval eval;

    float duration=eval.eval([&]() {
        cudaMemset(C.elements,0,dim*dim*sizeof(float));
        dim3 grid(dim/TILE_SIZE, dim/TILE_SIZE);
        dim3 block(TILE_SIZE,TILE_SIZE);
        matmul<<<grid,block>>>(A,B,C);
    });
    std::cout<<"Section: basic_tiled_matmul================================================="<<std::endl;
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
    std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    std::cout<<"isCalculationRight:"<<check(A,B,C)<<std::endl;
}

__global__ void matmul2(Matrix A, Matrix B, Matrix C) {
    __shared__ float tile_A[64][64];
    __shared__ float tile_B[64][64];


    float* matA = A.elements;
    float* matB = B.elements;
    float* matC = C.elements;

    int tile_id_row = blockIdx.y*64;
    int tile_id_col = blockIdx.x*64;
    int tile_offset_row = threadIdx.y*4;
    int tile_offset_col = threadIdx.x*4;
    float sum[16];

    #pragma unroll
    for(int i=0;i<16;i++){
        sum[i] = 0;
    }

    for(int i=0;i<gridDim.x;i++){
        for(int j=0;j<4;j++){
            int smem_row = tile_offset_row + j;
            int smem_col = tile_offset_col;

            int global_row_A = tile_id_row + smem_row;
            int global_col_A = i * 64 + smem_col;

            float4* vec_tileA = (float4*)&tile_A[smem_row][smem_col];
            float4* vecA = (float4*)&matA[global_col_A + global_row_A * A.width];
            *vec_tileA = *vecA;

            int global_row_B = i * 64 + smem_row;
            int global_col_B = tile_id_col + smem_col;

            float4* vec_tileB = (float4*)&tile_B[smem_row][smem_col];
            float4* vecB = (float4*)&matB[global_col_B + global_row_B * B.width];
            *vec_tileB = *vecB;
        }
        __syncthreads();
        for(int j=0;j<4;j++) {
            int a_row = tile_offset_row  + j;
            for (int t = 0; t < 64; t++) {
                float a = tile_A[a_row][t];
                float4 vecB = *(float4*)&tile_B[t][tile_offset_col];

                sum[j * 4 + 0] += a * vecB.x;
                sum[j * 4 + 1] += a * vecB.y;
                sum[j * 4 + 2] += a * vecB.z;
                sum[j * 4 + 3] += a * vecB.w;
            }
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++){
        int row = tile_id_row + tile_offset_row + i;
        int col = tile_id_col + tile_offset_col;
        float4* vecC = (float4*)&matC[row * C.width + col];
        *vecC = *(float4*)&sum[i * 4];
    }
}

void fragent_vec_tiled_matmul(int dim,Matrix A,Matrix B,Matrix C){
    Eval eval;
    float duration=eval.eval([&]() {
        dim3 grid(dim/64, dim/64);
        dim3 block(16,16);
        matmul2<<<grid,block>>>(A,B,C);
    });
    std::cout<<"Section:fragemnt_tiled_matmul================================================="<<std::endl;
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
    std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    std::cout<<"isCalculationRight:"<<check(A,B,C)<<std::endl;
}

__global__ void matmul3(Matrix A, Matrix B, Matrix C) {
    __shared__ float tile_A[64][64];
    __shared__ float tile_B[64][64];


    float* matA = A.elements;
    float* matB = B.elements;
    float* matC = C.elements;

    int tile_id_row = blockIdx.y*64;
    int tile_id_col = blockIdx.x*64;
    int tile_offset_row = threadIdx.y*4;
    int tile_offset_col = threadIdx.x*4;
    float sum[16];

    #pragma unroll
    for(int i=0;i<16;i++){
        sum[i] = 0;
    }

    for(int i=0;i<gridDim.x;i++){
        for(int j=0;j<4;j++){
            int smem_row = tile_offset_row + j;
            int smem_col = tile_offset_col;

            int global_row_A = tile_id_row + smem_row;
            int global_col_A = i * 64 + smem_col;

            float4* vec_tileA = (float4*)&tile_A[smem_row][smem_col];
            float4* vecA = (float4*)&matA[global_col_A + global_row_A * A.width];
            *vec_tileA = *vecA;

            int global_row_B = i * 64 + smem_row;
            int global_col_B = tile_id_col + smem_col;

            float4* vec_tileB = (float4*)&tile_B[smem_row][smem_col];
            float4* vecB = (float4*)&matB[global_col_B + global_row_B * B.width];
            *vec_tileB = *vecB;
        }
        __syncthreads();
        for(int j=0;j<16;j++){
            float4 sub_a[4];
            float4 sub_b[4];
            for(int k=0;k<4;k++){
                sub_a[k] = *(float4*)&tile_A[tile_offset_row+k][j*4];
                sub_b[k] = *(float4*)&tile_B[j*4+k][tile_offset_col];
            }
            for(int m=0;m<4;m++){
                sum[m*4+0] += sub_a[m].x*sub_b[0].x;
                sum[m*4+0] += sub_a[m].y*sub_b[1].x;
                sum[m*4+0] += sub_a[m].z*sub_b[2].x;
                sum[m*4+0] += sub_a[m].w*sub_b[3].x;
                sum[m*4+1] += sub_a[m].x*sub_b[0].y;
                sum[m*4+1] += sub_a[m].y*sub_b[1].y;
                sum[m*4+1] += sub_a[m].z*sub_b[2].y;
                sum[m*4+1] += sub_a[m].w*sub_b[3].y;
                sum[m*4+2] += sub_a[m].x*sub_b[0].z;
                sum[m*4+2] += sub_a[m].y*sub_b[1].z;
                sum[m*4+2] += sub_a[m].z*sub_b[2].z;
                sum[m*4+2] += sub_a[m].w*sub_b[3].z;
                sum[m*4+3] += sub_a[m].x*sub_b[0].w;
                sum[m*4+3] += sub_a[m].y*sub_b[1].w;
                sum[m*4+3] += sub_a[m].z*sub_b[2].w;
                sum[m*4+3] += sub_a[m].w*sub_b[3].w;
            }
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++){
        int row = tile_id_row + tile_offset_row + i;
        int col = tile_id_col + tile_offset_col;
        float4* vecC = (float4*)&matC[row * C.width + col];
        *vecC = *(float4*)&sum[i * 4];
    }
}

void useregister_tiled_matmul(int dim,Matrix A,Matrix B,Matrix C){
    Eval eval;
    float duration=eval.eval([&]() {
        dim3 grid(dim/64, dim/64);
        dim3 block(16,16);
        matmul3<<<grid,block>>>(A,B,C);
    });
    std::cout<<"Section:fragemnt_tiled_matmul================================================="<<std::endl;
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
    std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    std::cout<<"isCalculationRight:"<<check(A,B,C)<<std::endl;
}

__device__ __forceinline__ uint32_t smem_ptr_to_offset(void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__global__ void matmul4(Matrix A, Matrix B, Matrix C) {
    extern __shared__ float sdata[];
    float (*tile_A)[64][64] = (float(*)[64][64])sdata;
    float (*tile_B)[64][64] = (float(*)[64][64])&sdata[2*64*64];

    float* matA = A.elements;
    float* matB = B.elements;
    float* matC = C.elements;

    int tile_id_row = blockIdx.y*64;
    int tile_id_col = blockIdx.x*64;
    int tile_offset_row = threadIdx.y*4;
    int tile_offset_col = threadIdx.x*4;
    float sum[16];

    #pragma unroll
    for(int i=0;i<16;i++){
        sum[i] = 0;
    }

    // prologue first global to shared
    for(int j=0;j<4;j++){
        int i=0;
        int smem_row = tile_offset_row + j;
        int smem_col = tile_offset_col;

        int global_row_A = tile_id_row + smem_row;
        int global_col_A = i * 64 + smem_col;

        float4* vec_tileA = (float4*)&tile_A[0][smem_row][smem_col];
        float4* vecA = (float4*)&matA[global_col_A + global_row_A * A.width];
        uint32_t smem_ptr_A = smem_ptr_to_offset(vec_tileA);
        asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;\n"  :: "r"(smem_ptr_A), "l"(vecA), "n"(sizeof(float4)));

        int global_row_B = i * 64 + smem_row;
        int global_col_B = tile_id_col + smem_col;

        float4* vec_tileB = (float4*)&tile_B[0][smem_row][smem_col];
        float4* vecB = (float4*)&matB[global_col_B + global_row_B * B.width];
        uint32_t smem_ptr_B = smem_ptr_to_offset(vec_tileB);
        asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;\n"  :: "r"(smem_ptr_B), "l"(vecB), "n"(sizeof(float4)));
    }
    asm volatile ("cp.async.commit_group;\n" :: );

    for(int i=0;i<gridDim.x;i++){
        if(i<gridDim.x-1){
            int index = i+1;
            for(int j=0;j<4;j++){
                int smem_row = tile_offset_row + j;
                int smem_col = tile_offset_col;

                int global_row_A = tile_id_row + smem_row;
                int global_col_A = index * 64 + smem_col;

                float4* vec_tileA = (float4*)&tile_A[index%2][smem_row][smem_col];
                float4* vecA = (float4*)&matA[global_col_A + global_row_A * A.width];
                uint32_t smem_ptr_A = smem_ptr_to_offset(vec_tileA);
                asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;\n"  :: "r"(smem_ptr_A), "l"(vecA), "n"(sizeof(float4)));

                int global_row_B = index * 64 + smem_row;
                int global_col_B = tile_id_col + smem_col;

                float4* vec_tileB = (float4*)&tile_B[index%2][smem_row][smem_col];
                float4* vecB = (float4*)&matB[global_col_B + global_row_B * B.width];
                uint32_t smem_ptr_B = smem_ptr_to_offset(vec_tileB);
                asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;\n"  :: "r"(smem_ptr_B), "l"(vecB), "n"(sizeof(float4)));
            }
            asm volatile ("cp.async.commit_group;\n" :: );
        }
        
        if(i<gridDim.x-1){
            asm volatile ("cp.async.wait_group %0;\n" :: "n"(1));
        }
        else{
            asm volatile ("cp.async.wait_group %0;\n" :: "n"(0));
        }
        __syncthreads();
        for(int j=0;j<16;j++){
            float4 sub_a[4];
            float4 sub_b[4];
            for(int k=0;k<4;k++){
                sub_a[k] = *(float4*)&tile_A[i%2][tile_offset_row+k][j*4];
                sub_b[k] = *(float4*)&tile_B[i%2][j*4+k][tile_offset_col];
            }
            for(int m=0;m<4;m++){
                sum[m*4+0] += sub_a[m].x*sub_b[0].x;
                sum[m*4+0] += sub_a[m].y*sub_b[1].x;
                sum[m*4+0] += sub_a[m].z*sub_b[2].x;
                sum[m*4+0] += sub_a[m].w*sub_b[3].x;
                sum[m*4+1] += sub_a[m].x*sub_b[0].y;
                sum[m*4+1] += sub_a[m].y*sub_b[1].y;
                sum[m*4+1] += sub_a[m].z*sub_b[2].y;
                sum[m*4+1] += sub_a[m].w*sub_b[3].y;
                sum[m*4+2] += sub_a[m].x*sub_b[0].z;
                sum[m*4+2] += sub_a[m].y*sub_b[1].z;
                sum[m*4+2] += sub_a[m].z*sub_b[2].z;
                sum[m*4+2] += sub_a[m].w*sub_b[3].z;
                sum[m*4+3] += sub_a[m].x*sub_b[0].w;
                sum[m*4+3] += sub_a[m].y*sub_b[1].w;
                sum[m*4+3] += sub_a[m].z*sub_b[2].w;
                sum[m*4+3] += sub_a[m].w*sub_b[3].w;
            }
        }
    }
    for(int i=0;i<4;i++){
        int row = tile_id_row + tile_offset_row + i;
        int col = tile_id_col + tile_offset_col;
        float4* vecC = (float4*)&matC[row * C.width + col];
        *vecC = *(float4*)&sum[i * 4];
    }
}

void pipeline_tiled_matmul(int dim,Matrix A,Matrix B,Matrix C){
    cudaFuncSetAttribute(matmul4,cudaFuncAttributeMaxDynamicSharedMemorySize, 64 KB);
    Eval eval;
    float duration=eval.eval([&]() {
        dim3 grid(dim/64, dim/64);
        dim3 block(16,16);
        matmul4<<<grid,block,64 KB>>>(A,B,C);
    });
    std::cout<<"Section:fragemnt_tiled_matmul================================================="<<std::endl;
    std::cout<<"duration: "<<duration<<" ms"<<std::endl;
    std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
    std::cout<<"isCalculationRight:"<<check(A,B,C)<<std::endl;
}

int main(int argc,char* argv[]) {
    int dim =1024;
    if(argc>1) {
        dim = atoi(argv[1]);
    }
    assert(dim%128==0);

    Matrix A,B,C;
    initializeMatrix(&A, dim);
    initializeMatrix(&B, dim);
    initializeMatrix(&C, dim);
    basic_tiled_matmul(dim,A,B,C);
    fragent_vec_tiled_matmul(dim,A,B,C);
    useregister_tiled_matmul(dim,A,B,C);
    pipeline_tiled_matmul(dim,A,B,C);
}
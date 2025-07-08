#include "common/check.h"
#include <cublas_v2.h>
#include <cmath>

float check(Matrix A, Matrix B, Matrix result){
    Matrix C;
    int dim = A.width;
    initializeMatrix_zero(&C,dim);
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dim, dim, dim,
        &alpha,
        B.elements, dim,
        A.elements, dim,
        &beta,
        C.elements, dim
    );
    cublasDestroy(handle);

    float* arr_C = new float[dim*dim];
    float* arr_res = new float[dim*dim];
    cudaMemcpy(arr_C,C.elements, dim*dim*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(arr_res,result.elements, dim*dim*sizeof(float), cudaMemcpyDeviceToHost);
    float maxDiff = 0;
    for(int i=0;i<dim*dim;i++){
        float diff =std::abs(arr_C[i]-arr_res[i]);
        if(diff>maxDiff){
            maxDiff = diff;
        }
    }

    delete arr_C;
    delete arr_res;
    freeMatrix(&C);
    return maxDiff;
}
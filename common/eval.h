#pragma once
#include <cuda_runtime.h>
#include <functional>


class Eval{
    public:
        cudaEvent_t start, stop;
        Eval() {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }
        float eval(const std::function<void()>& fn){
            fn(); //warmup
            cudaDeviceSynchronize();                
            cudaEventRecord(start);
            for(int i=0;i<10;i++){
                fn();     
            }                             
            cudaEventRecord(stop);
            cudaEventSynchronize(stop); 
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            return ms/10;
        }
        ~Eval() {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
};
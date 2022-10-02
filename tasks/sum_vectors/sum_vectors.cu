#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>


#define ARR_LEN (536870912) // 536870912 for 4 GB, 134217728 for 1 GB, 13107200 for 100 MB, 131072 for 1MB
#define ARR_SIZE ((size_t)ARR_LEN * sizeof(double))

#define SAFE_CALL(err) do \
{ if (err != 0) \
    { \
     printf("ERROR [%s] in line %d: %s\n", __FILE__, \
     __LINE__, cudaGetErrorString(cudaGetLastError())); \
     exit(1); \
    } \
} while(0)


__global__ void calc_c_gpu (double* dC, double* dA, double* dB, size_t arr_len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  double sum = 0.;
  double ab = dA[i] * dB[i];
  
  if(i < arr_len) {
    for(int j = 0; j < 100; j++) {
        sum += sin(ab + (double)j);
    }
    
    dC[i] = sum;
  }
}


__global__ void init_ab_gpu (double* dA, double* dB, size_t arr_len) {  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < arr_len) {
    dA[i] = sin((double)i);
    dB[i] = cos(2*(double)i-5);
  }
}


void calc_c_cpu(double* C, double* A, double* B, size_t arr_len) {
    double sum, ab;
    for(size_t i = 0; i < arr_len; i++) 
    {
        sum = 0.;
        ab = A[i] * B[i];
  
        for(int j = 0; j < 100; j++) {
            sum += sin(ab + (double)j);
        }
    
        C[i] = sum;
    }
}


void init_ab_cpu(double* A, double* B, size_t arr_len) {
    for(size_t i = 0; i < arr_len; i++) 
    {
        A[i] = sin((double)i);
        B[i] = cos(2*(double)i-5);
    }
}


void compare_arrs(double* arr_1, double* arr_2, size_t arr_len)
{
    double max_err = 0.;
    size_t max_err_idx = 0;
    double avg_err = 0.;
    double accum_err = 0.;

    double diff = 0.;

    for(size_t i = 0; i < arr_len; i++)
    {
        diff = fabs(arr_1[i] - arr_2[i]);

        if (diff > max_err) {
            max_err = diff;
            max_err_idx = i;
        }

        accum_err += diff;
     }

    avg_err = accum_err / arr_len;

    printf("Max error idx = %lu: %lf -- %lf\n", max_err_idx, arr_1[max_err_idx],  arr_2[max_err_idx]);
    printf("Max error = %E\n", max_err);
    printf("Avg error = %E\n", avg_err);
}


int main(void)
{
    printf("Arr size: %lu\n", ARR_SIZE);
    printf("Arr len: %lu\n\n", ARR_LEN);
    
    float gpu_dth_time = 0.f, gpu_init_ab_time = 0.f, gpu_calc_c_time = 0.f, gpu_total_time = 0.f, gpu_calc_time = 0.f;
    float cpu_init_ab_time = 0.f, cpu_calc_c_time = 0.f, cpu_total_time = 0.f;

    cudaEvent_t start, stop; 
    SAFE_CALL (cudaEventCreate ( &start )); 
    SAFE_CALL (cudaEventCreate ( &stop ));

    // Allocate memory for device processing and copy to host
    double *hdC, *dC, *dA, *dB;
    SAFE_CALL (cudaMalloc ((void**) &dA, ARR_SIZE));
    SAFE_CALL (cudaMalloc ((void**) &dB, ARR_SIZE));
    SAFE_CALL (cudaMalloc ((void**) &dC, ARR_SIZE));
    SAFE_CALL (cudaMallocHost ((void**) &hdC, ARR_SIZE));

    // GPU A and B Initialization (+measure time)
    SAFE_CALL (cudaEventRecord ( start, 0));
    init_ab_gpu<<<(ARR_LEN + 1023) / 1024, 1024>>>(dA, dB, ARR_LEN);
    SAFE_CALL (cudaEventRecord ( stop, 0 ));

    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &gpu_init_ab_time, start, stop ));

    // GPU C calculation (+measure time)
    SAFE_CALL (cudaEventRecord ( start, 0));
    
    calc_c_gpu<<<(ARR_LEN + 1023) / 1024, 1024>>>(dC, dA, dB, ARR_LEN);

    SAFE_CALL (cudaEventRecord ( stop, 0 ));
    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &gpu_calc_c_time, start, stop ));

    // D2H (+measure time)
    SAFE_CALL (cudaEventRecord ( start, 0));

    SAFE_CALL (cudaMemcpy ( hdC, dC, ARR_SIZE, cudaMemcpyDeviceToHost ));
    
    SAFE_CALL (cudaEventRecord ( stop, 0 ));
    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &gpu_dth_time, start, stop ));

    // Free device memory
    SAFE_CALL (cudaFree(dA));
    SAFE_CALL (cudaFree(dB));
    SAFE_CALL (cudaFree(dC));

    // Allocate memory for host processing
    double* hA = (double*) malloc(ARR_SIZE);
    double* hB = (double*) malloc(ARR_SIZE);
    double* hC = (double*) malloc(ARR_SIZE);

    // CPU A and B Initialization (+measure time)
    SAFE_CALL (cudaEventRecord ( start, 0));

    init_ab_cpu(hA, hB, ARR_LEN);

    SAFE_CALL (cudaEventRecord ( stop, 0 ));
    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &cpu_init_ab_time, start, stop ));

    // CPU C calculation (+measure time)
    SAFE_CALL (cudaEventRecord ( start, 0));

    calc_c_cpu(hC, hA, hB, ARR_LEN);

    SAFE_CALL (cudaEventRecord ( stop, 0 ));
    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &cpu_calc_c_time, start, stop ));

    // Free hA, hB
    free(hA);
    free(hB);

    // Compare results from host and device
    compare_arrs(hC, hdC, ARR_LEN);

    // Print time consumption
    gpu_calc_time = gpu_init_ab_time + gpu_calc_c_time;
    gpu_total_time = gpu_calc_time + gpu_dth_time;
    cpu_total_time = cpu_init_ab_time + cpu_calc_c_time;

    printf ("\nGPU D2H memory copy time: %f ms\n", gpu_dth_time );
    printf ("GPU A and B initialization time: %f ms\n", gpu_init_ab_time );
    printf ("GPU C calculation time: %f ms\n", gpu_calc_c_time );
    printf ("GPU total calculation time time: %f ms\n", gpu_calc_time );
    printf ("GPU total time: %f ms\n\n", gpu_total_time );

    printf ("CPU A and B initialization time: %f ms\n", cpu_init_ab_time );
    printf ("CPU C calculation time: %f ms\n", cpu_calc_c_time );
    printf ("CPU total time: %f ms\n\n", cpu_total_time );

    printf("Acceleration (only calculation): %.1lfx (%.1lfx)\n\n", cpu_total_time / gpu_total_time, cpu_total_time / gpu_calc_time);
    
    // Free all allocated memory
    free(hC);
    cudaFreeHost(hdC);

    SAFE_CALL (cudaEventDestroy ( start ));
    SAFE_CALL (cudaEventDestroy ( stop ));

    std::cout << cudaGetErrorString(cudaGetLastError());
    return 0;
}

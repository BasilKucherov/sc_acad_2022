#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define ARR_LEN (104857600) // 536870912 for 4 GB, 134217728 for 1 GB, 104857600 for 100 MB
#define ARR_SIZE (ARR_LEN * sizeof(double))

#define SAFE_CALL(err) do \ 
{ if (err != 0) \
    { printf("ERROR [%s] in line %d: %s\n", __FILE__,
     __LINE__, cudaGetErrorString(err)); \
     exit(1); \ 
    }\
} while(0)


__global__ void do_math_gpu (double* dC, size_t arr_len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i < arr_len) {
    double sum = 0.;
    double ab = sin((double)i) * cos(2*(double)i-5);

    for(int j = 0; j < 100; j++) {
        sum += sin(ab + (double)j);
    }
    
    dC[i] = sum;
  }
}


void do_math_cpu(double* C, size_t arr_len) {
    double sum = 0;
    double ab = 0;

    for(int i = 0; i < arr_len; i++) {
        sum = 0;
        ab = sin((double)i) * cos(2*(double)i-5);

        for(int j = 0; j < 100; j++) {
            sum += sin(ab + (double)j);
        }

        C[i] = sum;
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
    printf("Max error = %.16lf\n", max_err);
    printf("Avg error = %.16lf\n", avg_err);
}


int main(void)
{
    printf("Arr size: %lu\n", ARR_SIZE);
    printf("Arr len: %lu\n\n", ARR_LEN);
    float cudamemcpy_dth_time = 0.;
    float calc_gpu_time = 0., calc_cpu_time = 0.;

    cudaEvent_t start, stop; 
    SAFE_CALL (cudaEventCreate ( &start )); 
    SAFE_CALL (cudaEventCreate ( &stop ));

    // Allocate memory for device processing and copy to host
    double *hdC, *dC;
    SAFE_CALL (cudaMalloc ((void**) &dC, ARR_SIZE));
    SAFE_CALL (cudaMallocHost ((void**) &hdC, ARR_SIZE));

    // Calculation on device (+measure time)
    SAFE_CALL (cudaEventRecord ( start, 0));
    do_math_gpu<<<(ARR_LEN + 1023) / 1024, 1024>>>(dC, ARR_LEN);
    SAFE_CALL (cudaEventRecord ( stop, 0 ));

    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &calc_gpu_time, start, stop ));


    // Copy D2H (+measure time)
    SAFE_CALL (cudaEventRecord ( start, 0));

    SAFE_CALL (cudaMemcpy ( hdC, dC, ARR_SIZE, cudaMemcpyDeviceToHost ));
    
    SAFE_CALL (cudaEventRecord ( stop, 0 ));
    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &cudamemcpy_dth_time, start, stop ));


    // Allocate memory for host processing
    double* hC = (double*) malloc(ARR_SIZE);

    // Calculation on host (+measure time)
    SAFE_CALL (cudaEventRecord ( start, 0));

    do_math_cpu(hC, ARR_LEN);

    SAFE_CALL (cudaEventRecord ( stop, 0 ));
    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &calc_cpu_time, start, stop ));

    // Compare results from host and device
    compare_arrs(hC, hdC, ARR_LEN);

    // Print time consumption
    printf ("\nGPU D2H memory copy time: %f ms\n", cudamemcpy_dth_time );
    printf ("GPU calculation time: %f ms\n\n", calc_gpu_time );
    printf ("GPU total time: %f ms\n",  cudamemcpy_dth_time + calc_gpu_time);

    printf ("CPU calculation time: %f ms\n", calc_cpu_time );

    printf("Acceleration (only calculation): %.1lfx (%.1lfx)\n", calc_cpu_time / (cudamemcpy_dth_time + calc_gpu_time), calc_cpu_time / calc_gpu_time);
    
    // Free all allocated memory
    free(hC);
    SAFE_CALL (cudaFree(hdC));
    SAFE_CALL (cudaFree(dC));
    SAFE_CALL (cudaEventDestroy ( start ));
    SAFE_CALL (cudaEventDestroy ( stop ));

    std::cout << cudaGetErrorString(cudaGetLastError());
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ARR_SIZE (4294967296) //   (104857600)
#define ARR_LEN (ARR_SIZE/sizeof(double))
#define CUDA_STREAM_NUMBER 1

__global__ void do_math_gpu (double* dC, size_t arr_len, size_t calc_offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int i_calc = i + calc_offset;

  if(i < arr_len) {
    double sum = 0.;
    double ab = sin((double)i_calc) * cos(2*(double)i_calc-5);

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
    float calc_gpu_time = 0.;

    cudaEvent_t start, stop; 
    cudaEventCreate ( &start ); 
    cudaEventCreate ( &stop );

    size_t arr_size_per_stream = ARR_SIZE / CUDA_STREAM_NUMBER;
    size_t arr_len_per_stream = ARR_LEN / CUDA_STREAM_NUMBER;	

	printf("size per stream: %lu\n\n", arr_size_per_stream);

    double *dC;
    cudaMalloc ((void**) &dC, ARR_SIZE);
    
    double *hC;
    cudaMallocHost ((void**) &hC, ARR_SIZE);

    cudaEventRecord ( start, 0);

    cudaStream_t streams[CUDA_STREAM_NUMBER];

    for(int i = 0; i < CUDA_STREAM_NUMBER; i++)
    {
        cudaStreamCreate ( &(streams[i]) );
    }

    for (int i = 0; i < CUDA_STREAM_NUMBER; i++ ) {
        do_math_gpu <<<(arr_len_per_stream + 1023) / 1024, 1024, 0, streams[i] >>> (dC + i * arr_len_per_stream, arr_len_per_stream, i * arr_len_per_stream);
    }

    for (int i = 0; i < CUDA_STREAM_NUMBER; i++ ) {
        cudaMemcpyAsync ( hC + i * arr_len_per_stream, dC + i * arr_len_per_stream, arr_size_per_stream, cudaMemcpyDeviceToHost, streams[i]);
    }

   // Синхронизация CUDA-streams
   cudaDeviceSynchronize ();
   // Уничтожение CUDA-streams
   for (int i = 0; i < CUDA_STREAM_NUMBER; i++ ) 
   {
	cudaStreamDestroy ( streams[i] );
   }

    cudaEventRecord ( stop, 0 );

    cudaEventSynchronize ( stop );
    cudaEventElapsedTime ( &calc_gpu_time, start, stop );

    printf ("GPU total time: %f ms\n",  calc_gpu_time);

    cudaFree(hC);
    cudaFree(dC);
    cudaEventDestroy ( start );
    cudaEventDestroy ( stop );
    return 0;
}

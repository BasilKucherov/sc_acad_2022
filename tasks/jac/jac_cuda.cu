#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define SAFE_CALL(err) do \
{ if (err != 0) \
    { \
     printf("ERROR [%s] in line %d: %s\n", __FILE__, \
     __LINE__, cudaGetErrorString(cudaGetLastError())); \
     exit(1); \
    } \
} while(0)


FILE *in;
int TRACE = 0;
int i, j, k, it;
double EPS;
int M, N, K, ITMAX;
double  MAXEPS = 0.1;
double time0;

double *A;
#define A(i,j,k) A[((i)*N+(j))*K+(k)]


double solution(int i, int j, int k)
{
    double x = 10.*i / (M - 1), y = 10.*j / (N - 1), z = 10.*k / (K - 1);
    return 2.*x*x - y*y - z*z;
    /*    return x+y+z; */
}

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps);


// Cuda kernels

__device__ double solution_gpu(int i, int j, int k, size_t x_dim, size_t y_dim, size_t z_dim)
{
    double x = 10.*i / (x_dim - 1), y = 10.*j / (y_dim - 1), z = 10.*k / (z_dim - 1);
    return 2.*x*x - y*y - z*z;
    /*    return x+y+z; */
}

__global__ void init_a_gpu (double* dA, size_t x_dim, size_t y_dim, size_t z_dim) {  
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;
  
  int global_idx = (i * y_dim + j) * z_dim + k;

  if (i < x_dim && j < y_dim && k < z_dim) {
    int is_boundary = ((i == 0) || (i == x_dim - 1) || (j == 0) || (j == y_dim - 1) || (k == 0) || (k == z_dim - 1));
    dA[global_idx] = is_boundary * solution_gpu(i, j, k, x_dim, y_dim, z_dim);
  }
}

#define dB(i,j,k) dB[((i)*y_dim+(j))*z_dim+(k)]
#define dA(i,j,k) dA[((i)*y_dim+(j))*z_dim+(k)]
#define dC(i,j,k) dC[((i)*y_dim+(j))*z_dim+(k)]


__global__ void jac_calc_b_gpu (double* dA, double* dB, size_t x_dim, size_t y_dim, size_t z_dim) {  
  int k = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  int i = 1 + blockIdx.z * blockDim.z + threadIdx.z;
  
  if (i < (x_dim - 1) && j < (y_dim-1) && k < (z_dim-1)) {
    dB(i, j, k) = (dA(i - 1, j, k) + dA(i + 1, j, k) + dA(i, j - 1, k) + dA(i, j + 1, k)
                                 + dA(i, j, k - 1) + dA(i, j, k + 1)) / 6.;
  }
}

__global__ void calc_c_gpu (double* dA, double* dB, double* dC, size_t x_dim, size_t y_dim, size_t z_dim) {  
  int k = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  int i = 1 + blockIdx.z * blockDim.z + threadIdx.z;
  
  if (i < (x_dim - 1) && j < (y_dim-1) && k < (z_dim-1)) {
    dC(i, j, k) = fabs(dB(i, j, k) - dA(i, j, k));
    dA(i, j, k) = dB(i, j, k);
  }
}

__global__ void max_reduction_gpu (double* dA, size_t arr_len, size_t step)
{
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * step * 2;

    if(idx + step < arr_len) {
        dA[idx] = Max(dA[idx], dA[idx + step]);
    }
}


double jac_gpu(double *dA, size_t x_dim, size_t y_dim, size_t z_dim, int itmax, double maxeps)
{
    double *dB, *dC;
    size_t dC_len = (x_dim) * (y_dim) * (z_dim);
    double eps;

    SAFE_CALL (cudaMalloc ((void**) &dB, x_dim * y_dim * z_dim * sizeof(double)));
    SAFE_CALL (cudaMalloc ((void**) &dC, dC_len * sizeof(double)));

    for (int it = 1; it <= itmax - 1; it++)
    {
        SAFE_CALL (cudaMemset(dC, 0, dC_len * sizeof(double)));

        jac_calc_b_gpu<<<dim3((z_dim + 7) / 8, (y_dim + 7) / 8, (x_dim + 7) / 8), dim3(8, 8, 8)>>>(dA, dB, x_dim, y_dim, z_dim);
        calc_c_gpu<<<dim3((z_dim + 7) / 8, (y_dim + 7) / 8, (x_dim + 7) / 8), dim3(8, 8, 8)>>>(dA, dB, dC, x_dim, y_dim, z_dim);

        size_t n_elems = (dC_len + 1) / 2;
        for(int step = 1; step < dC_len; step *= 2)
        {
            max_reduction_gpu<<<(n_elems + 1023) / 1024 , 1024>>>(dC, dC_len, step);
            n_elems = (n_elems + 1) / 2;
        }

        SAFE_CALL (cudaMemcpy ( &eps, dC, sizeof(double), cudaMemcpyDeviceToHost ));
        if (TRACE && it%TRACE == 0)
            printf("\nIT=%d eps=%.4g\t", it, eps);
        if (eps < maxeps) 
            break;
    }
    
    SAFE_CALL (cudaFree(dB));
    SAFE_CALL (cudaFree(dC));

    return eps;
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


int main(int an, char **as)
{
    // Getting parameters
    in = fopen("data3.in", "r");
    if (in == NULL) { printf("Can not open 'data3.in' "); exit(1); }
    i = fscanf(in, "%d %d %d %d %d", &M, &N, &K, &ITMAX, &TRACE);
    if (i < 4) 
    {
        printf("Wrong 'data3.in' (M N K ITMAX TRACE)");
        exit(2);
    }

    // Create cuda events for time measure
    cudaEvent_t start, stop; 
    SAFE_CALL (cudaEventCreate ( &start )); 
    SAFE_CALL (cudaEventCreate ( &stop ));

    float cpu_init_a_time = 0.f, cpu_calc_jac_time = 0.f, cpu_total_time = 0.f;
    float gpu_init_a_time = 0.f, gpu_calc_jac_time = 0.f, gpu_dth_time = 0.f, gpu_calc_time = 0.f, gpu_total_time = 0.f;


    // CPU Initializing A
    SAFE_CALL (cudaEventRecord ( start, 0));

    A = (double*) malloc(M*N*K*sizeof(double));

    for (i = 0; i <= M - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= K - 1; k++)
            {
                if (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1)
                    A(i, j, k) = solution(i, j, k);
                else 
                    A(i, j, k) = 0.;
            }

    SAFE_CALL (cudaEventRecord ( stop, 0 ));

    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &cpu_init_a_time, start, stop ));


    // GPU Initializing A
    SAFE_CALL (cudaEventRecord ( start, 0));

    size_t A_size = M*N*K*sizeof(double);
    double *dA;
    SAFE_CALL (cudaMalloc ((void**) &dA, A_size));

    init_a_gpu<<<dim3((K + 7) / 8, (N + 7) / 8, (M + 7) / 8), dim3(8, 8, 8)>>>(dA, M, N, K);

    SAFE_CALL (cudaEventRecord ( stop, 0 ));

    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &gpu_init_a_time, start, stop ));

    // Compare A GPU and A CPU
    double *hdA;
    SAFE_CALL (cudaMallocHost ((void**) &hdA, A_size));
    SAFE_CALL (cudaMemcpy ( hdA, dA, A_size, cudaMemcpyDeviceToHost ));

    printf("compare A arrs after initialization:\n");
    compare_arrs(hdA, A, M*N*K);

    // GPU Calculation
    double gpu_eps = 0.;
    SAFE_CALL (cudaEventRecord ( start, 0));
    gpu_eps = jac_gpu(dA, M, N, K, ITMAX, MAXEPS);
    SAFE_CALL (cudaEventRecord ( stop, 0 ));

    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &gpu_calc_jac_time, start, stop ));
    
    // D2H
    SAFE_CALL (cudaEventRecord ( start, 0));

    SAFE_CALL (cudaMemcpy ( hdA, dA, A_size, cudaMemcpyDeviceToHost ));

    SAFE_CALL (cudaEventRecord ( stop, 0 ));

    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &gpu_dth_time, start, stop ));

    SAFE_CALL(cudaFree(dA));


    // CPU Calculation
    SAFE_CALL (cudaEventRecord ( start, 0));

    EPS = jac(A, M, N, K, ITMAX, MAXEPS);   
    
    SAFE_CALL (cudaEventRecord ( stop, 0 ));

    SAFE_CALL (cudaEventSynchronize ( stop ));
    SAFE_CALL (cudaEventElapsedTime ( &cpu_calc_jac_time, start, stop ));

    printf("\n\n---- RESULTS ----\n\n");
    printf("CPU EPS: %lf\n", EPS);
    printf("GPU EPS: %lf\n", gpu_eps);
    printf("err: %E\n\n", fabs(EPS - gpu_eps));
    printf("compare A arrs:\n");
    compare_arrs(hdA, A, M*N*K);


    // printf("%dx%dx%d x %d\t<", M, N, K, ITMAX);
    // printf("%3.1f>\teps=%.4g ", time0, EPS);

    // if (TRACE)
    // {
    //     EPS = 0.;

    //     for (i = 0; i <= M - 1; i++)
    //         for (j = 0; j <= N - 1; j++)
    //             for (k = 0; k <= K - 1; k++)
    //                 EPS = Max(fabs(A(i, j, k) - solution(i, j, k)), EPS);
    //     printf("delta=%.4g\n", EPS);
    // }

    // Print time consumption
    cpu_total_time = cpu_init_a_time + cpu_calc_jac_time;
    gpu_calc_time = gpu_init_a_time + gpu_calc_jac_time;
    gpu_total_time = gpu_calc_time + gpu_dth_time;

    printf ("\nCPU A initialization time: %f ms\n", cpu_init_a_time );
    printf ("CPU JAC calculation time: %f ms\n", cpu_calc_jac_time );
    printf ("CPU total time: %f ms\n", cpu_total_time );

    printf ("\n\nGPU A initialization time: %f ms\n", gpu_init_a_time );
    printf ("GPU JAC calculation time: %f ms\n", gpu_calc_jac_time );
    printf ("GPU D2H time: %f ms\n", gpu_dth_time );
    printf ("GPU calculation time: %f ms\n", gpu_calc_time );
    printf ("GPU total time: %f ms\n", gpu_total_time );

    printf("\nAcceleration (only calculation): %.1lfx (%.1lfx)\n\n", cpu_total_time / gpu_total_time, cpu_total_time / gpu_calc_time);

    // Free all allocated memory
    free(A);
    SAFE_CALL (cudaFreeHost(hdA));

    std::cout << cudaGetErrorString(cudaGetLastError());

    return 0;
}

#define a(i,j,k) a[((i)*nn+(j))*kk+(k)]
#define b(i,j,k) b[((i)*nn+(j))*kk+(k)]

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps)
{
    double *b;
    int i, j, k;
    double eps;

    b = (double*) malloc(mm*nn*kk*sizeof(double));

    for (it = 1; it <= itmax - 1; it++)
    {
        for (i = 1; i <= mm - 2; i++)
            for (j = 1; j <= nn - 2; j++)
                for (k = 1; k <= kk - 2; k++)
                    b(i, j, k) = (a(i - 1, j, k) + a(i + 1, j, k) + a(i, j - 1, k) + a(i, j + 1, k)
                                 + a(i, j, k - 1) + a(i, j, k + 1)) / 6.;

        eps = 0.;
        for (i = 1; i <= mm - 2; i++)
            for (j = 1; j <= nn - 2; j++)
                for (k = 1; k <= kk - 2; k++)
                {
                    eps = Max(fabs(b(i, j, k) - a(i, j, k)), eps);
                    a(i, j, k) = b(i, j, k);
                }

        if (TRACE && it%TRACE == 0)
            printf("\nIT=%d eps=%.4g\t", it, eps);
        if (eps < maxeps) 
            break;
    }
    free(b);
    return eps;
}

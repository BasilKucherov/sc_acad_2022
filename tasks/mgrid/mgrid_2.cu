#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
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
int     M, N, K, ITMAX;
double  MAXEPS = 0.1;
double time0;

double *A;
#define A(i,j,k) A[((i)*N+(j))*K+(k)]

// GPU
#define dB(i,j,k) dB[((i)*y_dim+(j))*z_dim+(k)]
#define dA(i,j,k) dA[((i)*y_dim+(j))*z_dim+(k)]
#define dA2(i,j,k) dA2[((i)*y_dim2+(j))*z_dim2+(k)]
#define dC(i,j,k) dC[((i)*y_dim+(j))*z_dim+(k)]

__device__ double solution_gpu(int i, int j, int k, size_t x_dim, size_t y_dim, size_t z_dim)
{
    double x = 10.*i / (x_dim - 1), y = 10.*j / (y_dim - 1), z = 10.*k / (z_dim - 1);
    return 2.*x*x - y*y - z*z;
    /*    return x+y+z; */
}

__global__ void init_a_gpu (double* dA, int x_dim, int y_dim, int z_dim) {  
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int i = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (i < x_dim && j < y_dim && k < z_dim) {
    int is_boundary = ((i == 0) || (i == x_dim - 1) || (j == 0) || (j == y_dim - 1) || (k == 0) || (k == z_dim - 1));
    dA(i, j, k) = is_boundary * solution_gpu(i, j, k, x_dim, y_dim, z_dim);
  }
}


__global__ void calc_a2_gpu(double *dA2, double *dA, size_t x_dim, size_t y_dim, size_t z_dim, size_t x_dim2, size_t y_dim2, size_t z_dim2)
{
  size_t k2 = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j2 = blockIdx.y * blockDim.y + threadIdx.y;
  size_t i2 = blockIdx.z * blockDim.z + threadIdx.z;


  if (i2 < x_dim2 && j2 < y_dim2 && k2 < z_dim2)
  { 
    
    size_t i_res, j_res, k_res;
    int res = 0;
    for(int i = i2 * 2; i <= i2 * 2 + 1; i++)
        for(int j = j2 * 2; j <= j2 * 2 + 1; j++)
            for(int k = k2 * 2; k <= k2 * 2 + 1; k++)
                if ((i % 2 == 0 || i == x_dim - 1) && (j % 2 == 0 || j == y_dim - 1) && (k % 2 == 0 || k == z_dim - 1))
                {
                    res = 1;
                    i_res = i;
                    j_res = j;
                    k_res = k;
                }                    
    
    dA2(i2, j2, k2) = dA(i_res, j_res, k_res) * res;
  }
}

__global__ void calc_a_gpu(double *dA2, double *dA, size_t x_dim, size_t y_dim, size_t z_dim, size_t x_dim2, size_t y_dim2, size_t z_dim2)
{
  size_t k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t i = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i < x_dim - 1 && j < y_dim - 1 && k < z_dim - 1)
  {
    dA(i, j, k) = (
        dA2(i / 2, j / 2, k / 2) +
        dA2(i / 2, j / 2, k / 2 + k % 2) +
        dA2(i / 2, j / 2 + j % 2, k / 2) +
        dA2(i / 2, j / 2 + j % 2, k / 2 + k % 2) +
        dA2(i / 2 + i % 2, j / 2, k / 2) +
        dA2(i / 2 + i % 2, j / 2, k / 2 + k % 2) +
        dA2(i / 2 + i % 2, j / 2 + j % 2, k / 2) +
        dA2(i / 2 + i % 2, j / 2 + j % 2, k / 2 + k % 2)
        ) / 8.;
  }
}

// JAC
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

    for (int it = 1; it <= itmax; it++)
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

// JAC end



void compare_arrs(double* arr_1, double* arr_2, size_t arr_len)
{
    double max_err = 0.;
    size_t max_err_idx = 0;
    double avg_err = 0.;
    double accum_err = 0.;
    size_t number_exc = 0;

    double diff = 0.;

    for(size_t i = 0; i < arr_len; i++)
    {
        diff = fabs(arr_1[i] - arr_2[i]);

        if (diff > max_err) {
            max_err = diff;
            max_err_idx = i;
        }

        if (diff > 0.00000001) {
            number_exc += 1;
            // printf("[%d] = %lf %lf\n", i, arr_1[i], arr_2[i]);
        }

        
        // printf("[%lu] = %lf %lf\n", i, arr_1[i], arr_2[i]);

        accum_err += diff;
     }

    avg_err = accum_err / arr_len;

    printf("Max error idx = %lu: %lf -- %lf\n", max_err_idx, arr_1[max_err_idx],  arr_2[max_err_idx]);
    printf("Max error = %E\n", max_err);
    printf("Avg error = %E\n", avg_err);
    printf("Number exceed = %lu / %lu\n", number_exc, arr_len);
}


//


double solution(int i, int j, int k)
{
    double x = 10.*i / (M - 1), y = 10.*j / (N - 1), z = 10.*k / (K - 1);
    return 2.*x*x - y*y - z*z;
}

double jac(double *a, double *dA, int mm, int nn, int kk, int itmax, double maxeps);

int main(int an, char **as)
{
    in = fopen("data3.in", "r");
    if (in == NULL) { printf("Can not open 'data3.in' "); exit(1); }
    i = fscanf(in, "%d %d %d %d %d", &M, &N, &K, &ITMAX, &TRACE);
    if (i<4) 
    {
        printf("Wrong 'data3.in' (M N K ITMAX TRACE)");
        exit(2);
    }


    // GPU

    double *dA, *hA;
    size_t A_len = M * N * K;
    size_t A_size = A_len * sizeof(double);
    SAFE_CALL (cudaMalloc((void**) &dA, A_size));
    SAFE_CALL (cudaMallocHost((void**) &hA, A_size));

    dim3 block(8,8,8);
    dim3 grid((K + 7) / 8, (N + 7) / 8, (M + 7) / 8);
    init_a_gpu<<<grid, block>>>(dA, M, N, K);

    //


    A = (double*) malloc(M*N*K * sizeof(double));

    for (i = 0; i <= M - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= K - 1; k++)
            {
                if (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1)
                    A(i, j, k) = solution(i, j, k);
                else 
                    A(i, j, k) = 0.;
            }


    // Compare A arrs after init

    SAFE_CALL (cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost));
    printf("------------------------\n");
    printf("Compare A arrs after init:\n");
    compare_arrs(A, hA, A_len);
    printf("\n");
    printf("------------------------\n");

    // 


    printf("%dx%dx%d x %d\t<", M, N, K, ITMAX);
    time0 = 0.;
    EPS = jac(A, dA, M, N, K, ITMAX, MAXEPS);    
    printf("%3.1f>\teps=%.4g ", time0, EPS);
    
    if (TRACE)
    {
        EPS = 0.;
        for (i = 0; i <= M - 1; i++)
            for (j = 0; j <= N - 1; j++)
                for (k = 0; k <= K - 1; k++)
                    EPS = Max(fabs(A(i, j, k) - solution(i, j, k)), EPS);
        printf("delta=%.4g\n", EPS);
    }

    free(A);
    SAFE_CALL (cudaFree(dA));
    SAFE_CALL (cudaFreeHost(hA));

    std::cout << cudaGetErrorString(cudaGetLastError());

    return 0;
}

#define a(i,j,k) a[((i)*nn+(j))*kk+(k)]
#define b(i,j,k) b[((i)*nn+(j))*kk+(k)]
#define a2(i,j,k) a2[((i)*n2+(j))*k2+(k)]
#define a2_mod(i,j,k) a2_mod[((i)*n2+(j))*k2+(k)]

double jac(double *a, double *dA, int mm, int nn, int kk, int itmax, double maxeps)
{


    // GPU
    double gpu_eps;
    size_t x_dim = mm, y_dim = nn, z_dim = kk;
    size_t dA_len = x_dim * y_dim * z_dim;
    size_t dA_size = dA_len * sizeof(double);

    double *hA;
    SAFE_CALL (cudaMallocHost((void**) &hA, dA_size));
    //


    double *b;
    int i, j, k, it;
    double eps;

    if (mm > 31 && nn > 31)
    {


        // GPU

        size_t x_dim2 = (x_dim + 1) / 2;
        size_t y_dim2 = (y_dim + 1) / 2;
        size_t z_dim2 = (z_dim + 1) / 2;

        double *dA2;
        size_t dA2_len = x_dim2 * y_dim2 * z_dim2;
        size_t dA2_size = dA2_len * sizeof(double);

        SAFE_CALL (cudaMalloc((void**) &dA2, dA2_size));
        SAFE_CALL (cudaMemset(dA2, 0, dA2_size));


        dim3 block(8, 8, 8);
        dim3 grid((z_dim2 + 7) / 8, (y_dim2 + 7) / 8, (x_dim2 + 7) / 8);
        calc_a2_gpu<<<grid, block>>>(dA2, dA, x_dim, y_dim, z_dim, x_dim2, y_dim2, z_dim2);

        //


        int m2 = (mm + 1) / 2, n2 = (nn + 1) / 2, k2 = (kk + 1) / 2;
        double *a2;
        a2 = (double*) malloc(m2*n2*k2 * sizeof(double));
        memset(a2, 0, m2*n2*k2 * sizeof(double));

  
        for (i = 0; i <= mm - 1; i++)
            for (j = 0; j <= nn - 1; j++)
                for (k = 0; k <= kk - 1; k++)                
                    if ((i % 2 == 0 || i == mm - 1) && (j % 2 == 0 || j == nn - 1) && (k % 2 == 0 || k == kk - 1))
                        a2(i / 2, j / 2, k / 2) = a(i, j, k);


        // Compare A2 after calc a2

        SAFE_CALL (cudaMemcpy(hA, dA2, dA2_size, cudaMemcpyDeviceToHost));
        printf("\n\n------------------------\n");
        printf("Compare A2 arrs after calc a2:\n");
        compare_arrs(a2, hA, dA2_len);
        printf("\n");
        printf("------------------------\n\n");


        eps = jac(a2, dA2, m2, n2, k2, itmax * 2, maxeps);
        

        // Compare all

        SAFE_CALL (cudaMemcpy(hA, dA2, dA2_size, cudaMemcpyDeviceToHost));
        printf("\n\n------------------------\n");
        printf("Compare A2 arrs before calc a:\n");
        compare_arrs(a2, hA, dA2_len);
        printf("\n");
        printf("------------------------\n");

        SAFE_CALL (cudaMemcpy(hA, dA, dA_size, cudaMemcpyDeviceToHost));
        printf("\n------------------------\n");
        printf("Compare A arrs before calc a:\n");
        compare_arrs(a, hA, dA_len);
        printf("\n");
        printf("------------------------\n\n");

        // 


        // GPU

        grid = dim3((z_dim + 5) / 8, (y_dim + 5) / 8, (x_dim + 5) / 8);
        calc_a_gpu<<<grid, block>>>(dA2, dA, x_dim, y_dim, z_dim, x_dim2, y_dim2, z_dim2);

        //


        for (i = 1; i <= mm - 2; i++)
            for (j = 1; j <= nn - 2; j++)
                for (k = 1; k <= kk - 2; k++)
                {
                    a(i, j, k) = (
                        a2(i / 2, j / 2, k / 2) +
                        a2(i / 2, j / 2, k / 2 + k % 2) +
                        a2(i / 2, j / 2 + j % 2, k / 2) +
                        a2(i / 2, j / 2 + j % 2, k / 2 + k % 2) +
                        a2(i / 2 + i % 2, j / 2, k / 2) +
                        a2(i / 2 + i % 2, j / 2, k / 2 + k % 2) +
                        a2(i / 2 + i % 2, j / 2 + j % 2, k / 2) +
                        a2(i / 2 + i % 2, j / 2 + j % 2, k / 2 + k % 2)
                        ) / 8.;
                }
        
        // Compare A arrs after calc a      

        SAFE_CALL (cudaMemcpy(hA, dA, dA_size, cudaMemcpyDeviceToHost));
        printf("\n\n------------------------\n");
        printf("Compare A arrs after calc a:\n");
        compare_arrs(a, hA, dA_len);
        printf("\n");
        printf("------------------------\n\n");

        // 

        free(a2);
        SAFE_CALL (cudaFree(dA2));
    }


    // GPU
    // Compare A before Jac VSPYSHKA   

    SAFE_CALL (cudaMemcpy(hA, dA, dA_size, cudaMemcpyDeviceToHost));
    printf("\n\n------------------------\n");
    printf("Compare A arrs before Jac VSPYSHKA   :\n");
    compare_arrs(a, hA, dA_len);
    printf("\n");
    printf("------------------------\n\n");

    // 

    
    gpu_eps = jac_gpu(dA, x_dim, y_dim, z_dim, itmax, maxeps);


    //


    b = (double*) malloc(mm*nn*kk * sizeof(double));

    for (it = 1; it <= itmax; it++)
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

        if (TRACE && it % TRACE == 0) 
            printf("\nIT=%d eps=%.4g\t", it, eps);
        if (eps < maxeps) 
            break;
    }

    // Compare A after Jac VSPYSHKA   

    SAFE_CALL (cudaMemcpy(hA, dA, dA_size, cudaMemcpyDeviceToHost));
    printf("\n\n------------------------\n");
    printf("Compare A arrs after Jac VSPYSHKA   :\n");
    compare_arrs(a, hA, dA_len);
    printf("\n");
    printf("cpu vs gpu: %.4g %.4g\n", eps, gpu_eps);
    printf("------------------------\n\n");

    // 

    free(b);
    SAFE_CALL (cudaFreeHost(hA));

    return eps;
}
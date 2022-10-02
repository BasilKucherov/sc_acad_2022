#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))


FILE *in;
int TRACE = 0;
int i, j, k, it;
double EPS;
int     M, N, K, ITMAX;
double  MAXEPS = 0.1;
double time0;

double *A;
#define A(i,j,k) A[((i)*N+(j))*K+(k)]

double solution(int i, int j, int k)
{
    double x = 10.*i / (M - 1), y = 10.*j / (N - 1), z = 10.*k / (K - 1);
    return 2.*x*x - y*y - z*z;
}

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps);

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

    A = malloc(M*N*K * sizeof(double));

    for (i = 0; i <= M - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= K - 1; k++)
            {
                if (i == 0 || i == M - 1 || j == 0 || j == N - 1 || k == 0 || k == K - 1)
                    A(i, j, k) = solution(i, j, k);
                else 
                    A(i, j, k) = 0.;
            }


    printf("%dx%dx%d x %d\t<", M, N, K, ITMAX);
    time0 = 0.;
    EPS = jac(A, M, N, K, ITMAX, MAXEPS);    
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
    return 0;
}

#define a(i,j,k) a[((i)*nn+(j))*kk+(k)]
#define b(i,j,k) b[((i)*nn+(j))*kk+(k)]
#define a2(i,j,k) a2[((i)*n2+(j))*k2+(k)]

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps)
{
    double *b;
    int i, j, k, it;
    double eps;

    if (mm > 31 && nn > 31)
    {
        int m2 = (mm + 1) / 2, n2 = (nn + 1) / 2, k2 = (kk + 1) / 2;
        double *a2;
        a2 = malloc(m2*n2*k2 * sizeof(double));

        for (i = 0; i <= mm - 1; i++)
            for (j = 0; j <= nn - 1; j++)
                for (k = 0; k <= kk - 1; k++)                
                    if ((i % 2 == 0 || i == mm - 1) && (j % 2 == 0 || j == nn - 1) && (k % 2 == 0 || k == kk - 1))
                        a2(i / 2, j / 2, k / 2) = a(i, j, k);
        eps = jac(a2, m2, n2, k2, itmax * 2, maxeps);
        
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

        free(a2);
    }
    b = malloc(mm*nn*kk * sizeof(double));

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
    free(b);
    return eps;
}









__global__ void calc_a2_gpu (double* dA, double* dA2, double* dC, size_t x_dim, size_t y_dim, size_t z_dim) {  
  int k = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  int i = 1 + blockIdx.z * blockDim.z + threadIdx.z;
  
  if (i < (x_dim - 1) && j < (y_dim-1) && k < (z_dim-1)) {
    int cond = ((i % 2 == 0 || i == x_dim - 1) && (j % 2 == 0 || j == y_dim - 1) && (k % 2 == 0 || k == z_dim - 1))
    dA2(i,j,k) = cond * dA(i, j, k);
  }
}
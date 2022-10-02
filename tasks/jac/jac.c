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
    /*    return x+y+z; */
}

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps);

int main(int an, char **as)
{
    in = fopen("data3.in", "r");
    if (in == NULL) { printf("Can not open 'data3.in' "); exit(1); }
    i = fscanf(in, "%d %d %d %d %d", &M, &N, &K, &ITMAX, &TRACE);
    if (i < 4) 
    {
        printf("Wrong 'data3.in' (M N K ITMAX TRACE)");
        exit(2);
    }

    A = malloc(M*N*K*sizeof(double));

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

double jac(double *a, int mm, int nn, int kk, int itmax, double maxeps)
{
    double *b;
    int i, j, k;
    double eps;

    b = malloc(mm*nn*kk*sizeof(double));

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

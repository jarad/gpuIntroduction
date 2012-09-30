#include <Rmath.h>
//#include <stdlib.h>


int cpu_runif(int n, double ub, int ni, int nd, double *u, int *count)
{
    int i, j, a;
    double b;
    GetRNGstate();
    for (i=0;i<n;i++) 
    {
        count[i] = -1;
        u[i] = ub+1;
        while ( u[i]>ub  ) 
        {
            count[i]++;
            //u[i] = rand()/((double)RAND_MAX + 1);
            u[i] = runif(0,1);

            // Computational overhead
            a=0; for (j=0; j<ni; j++) a += 1;
            b=1; for (j=0; j<nd; j++) b *= 1.00001;
        }
    }
    PutRNGstate();
}

void cpu_runif_wrap(int *n, double *ub, int *ni, int *nd, double *u, int *count)
{
    cpu_runif(*n, *ub, *ni, *nd, u, count);
}


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WARN 1
#define THRESH 1e-5
#define MAX_ITER 10000
#define STEP 0.1

#define EULER 0.57721566
#define MAXIT 100
#define FPMIN 1.0e-30
#define EPS 6.0e-8

#define MAX(a,b) (((a)>(b))?(a):(b))
#define NFACTOR 20
#define MIN_REC 1e-12

double Ne_t(double a, double V, int N) {
    double prod_sum = 0, Ne_sum = 0;
		double Qt;
    int t=0, niter=0;
    double reldiff = INFINITY, last=N;
    while ((reldiff >= THRESH) & (niter < MAX_ITER)) {
        /* Q_ta += pow(1-k, t) * pow(1-rf, t); */
        Qt = (1-pow(a, t+1)) / (1-a);
				double Ne = N*exp(-V/2 * pow(Qt, 2));
        prod_sum += log(1-0.5/Ne);
        Ne_sum += exp(prod_sum); 
        /* printf("Qt = %g\n", Qt); */
        /* printf("%d Ne_sum = %g, Ne = %g, abs diff = %g\t", niter, Ne_sum, Ne, fabs(Ne_sum - last)); */
        reldiff = fabs(Ne_sum - last) / Ne_sum;
        last = Ne_sum;
        /* printf("rel diff: %g, niter: %d\n", reldiff, niter); */
        /* printf("cond = %d, thresh = %g\n", reldiff >= THRESH, THRESH); */
        niter++;
        t++;
        /* printf("rel diff: %g, niter: %d\n", reldiff, niter); */
		}
    if (niter > MAX_ITER)
        printf("WARNING: Ne_t did not converge!");
 		return Ne_sum / 2;
}

double Ne_t_rescaled(double a, double V, int N, double scale) {
    double Q, res = 0;
    int niter = 0;
    for (int z=0; z<=log(10*N+1); z++) {
        double sum = 0, last = 0;
        for (double i=0; i<= (exp(z)-1)/N + scale; i+=scale) {
            Q = (1-pow(a, (N*i)+1)) / (1-a);
            double val = exp(0.5*V * pow(Q, 2));
            if (i > 0) {
                sum += scale*N * 0.5*(val + last);
                /* printf("trap: %f - %f \n", val, last); */
            }
            last = val;
            niter++;
        }
        if (niter > MAX_ITER) break;
        res += exp(z - 0.5/N * sum);
    }
    /* printf("niter: %d\n", niter); */
    if (niter > MAX_ITER)
        printf("WARNING: Ne_t_rescaled did not converge!");
    /* printf("niter: %d, end: %f\n", niter, log(10*N+1)); */
    return res/2;
}

double Ne_t_rescaled2(double a, double V, int N, double scale) {
    // In development
    double Q, res = 0;
    int niter = 0;
    for (int z=0; z<=log(10*N+1); z++) {
        double sum = 0, last = 0;
        double step;
        for (double i=0; i<= log(exp(z)-1); i++) {
            step = pow(2, i);
            Q = (1-pow(a, exp(i)+1)) / (1-a);
            double val = exp(0.5*V * pow(Q, 2));
            if (i > 0) {
                sum += (pow(2, i+1) - pow(2, i)) * 0.5*(val + last);
                printf("i=%d, step=%f, trap: %f - %f \n", i, step, val, last);
            }
            last = val;
            niter++;
        }
        if (niter > MAX_ITER) break;
        res += exp(z - 0.5/N * sum);
    }
    /* printf("niter: %d\n", niter); */
    if (niter > MAX_ITER)
        printf("WARNING: Ne_t_rescaled did not converge!");
    printf("niter: %d, end: %f\n", niter, log(10*N+1));
    return res/2;
}

void B_BK2022(const double *a, const double *V, 
              double *B, ssize_t n, const int N, 
              const double scaling) {
    // Our version of S&C '16's equation
		assert(a > 0);
		assert(a < 1);
    if (scaling < 0) {
        /* printf("note: asymptotic Ne mode.\n"); */
        // asymptotic Ne results
        for (ssize_t i=0; i<n; i++) {
            double Q = 1 / (1-a[i]);
            B[i] = exp(-V[i]/2 * pow(Q, 2));
            /* printf("%g, ", B[i]); */
            /* printf("."); */
        }
    } else if (scaling == 0) {
        /* printf("note: exact Ne mode.\n"); */
        // exact
        for (ssize_t i=0; i<n; i++) {
            B[i] = Ne_t(a[i], V[i], N) / N; 
            /* printf("."); */
        }
    } else {
        // rescaled
        /* printf("note: exact Ne mode (rescaled).\n"); */
        for (ssize_t i=0; i<n; i++) {
            B[i] = Ne_t_rescaled(a[i], V[i], N, scaling) / N; 
            /* printf("."); */
        }
    }
}


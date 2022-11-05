#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sum.h>

#define WARN 1
#define THRESH 1e-5

#define EULER 0.57721566
#define MAXIT 100
#define FPMIN 1.0e-30
#define EPS 6.0e-8

#define MAX(a,b) (((a)>(b))?(a):(b))
#define NFACTOR 20
#define MIN_REC 1e-12

double g(double Ne_t) {
		double log_prod = 0;
		double Ne_asymp;
		for (int t=1; t < T; t++) {
        /* Q_ta += pow(1-k, t) * pow(1-rf, t); */
        Q_t = (1-pow(a, t+1)) / (1-a);
				double Ne = N*exp(-V/2 * pow(Q_t, 2)); // factor of two needed?
        log_prod = log_prod + log(1-0.5/Ne);
				double f2 = 


        /* prod *= (1-0.5/Ne); */
        /* printf("Ne = %f, exp(log prod) = %f, Q_t = %f, Q_ta = %f\n", Ne, exp(log_prod), Q_t, Q_ta); */
        Ne_t[t] = Ne_t[t-1] + exp(log_prod);
 
}

double B_BK2022(double V, double Vm, double rf, int N) {
    // Our version of S&C '16's equation
    double k = Vm/V;
    double Q2;
    int T = 10000;
    double Ne_t[T];
		/* double prod = 1; */
    assert(k > 0);
    assert(k < 1);
    assert(Vm > 0);
    assert(V > 0);

		/* double sum_accel, err; */
		/* gsl_sum_levin_u_workspace *w = gsl_sum_levin_u_alloc(T); */

    double Q_t = 1;
    double Q_ta = 1;
		/* gsl_sum_levin_u_accel(Ne_t, T, w, &sum_accel, &err); */
		Ne_t[0] = 0;
		double last_f = 0;
    double a = (1-k) * (1-rf);
    int step = 10;
       /* printf("Ne(%d) = %f, diff = %g\n", t, Ne_t[t], Ne_t[t] - Ne_t[t-1]); */
        printf("%f,%g\n", Ne_t[t], Ne_t[t] - Ne_t[t-1]);
    }
    /* printf("==========================DONE\n"); */ 
    /* gsl_sum_levin_u_accel(Ne_t, T, w, &sum_accel, &err); */
    /* printf("Ne_[-1]=%g, sum accel = %g", Ne_t[T], sum_accel); */
		/* printf("estimated error  = % .16f\n", err); */
  	/* gsl_sum_levin_u_free(w); */
    return 0.5*Ne_t[T-1]/N;
}




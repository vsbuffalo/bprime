#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_sf_expint.h>

#define EULER 0.57721566
#define MAXIT 100
#define FPMIN 1.0e-30
#define EPS 6.0e-8

#define MAX(a,b) (((a)>(b))?(a):(b))
#define NFACTOR 3
#define MIN_REC 1e-12

void nrerror(char error_text[])
		/* Numerical Recipes standard error handler */
{
		fprintf(stderr,"Numerical Recipes run-time error...\n");
		fprintf(stderr,"%s\n",error_text);
		fprintf(stderr,"...now exiting to system...\n");
		exit(1);
}

double ei(double x)
{
		void nrerror(char error_text[]);
		int k;
		double fact,prev,sum,term;
    printf("x=%g", x);
		if (x <= 0.0) nrerror("Bad argument in ei");
		if (x < FPMIN) return log(x)+EULER;
		if (x <= -log(EPS)) {
				sum=0.0;
				fact=1.0;
				for (k=1;k<=MAXIT;k++) {
						fact *= x/k;
						term=fact/k;
						sum += term;
						if (term < EPS*sum) break;
				}
				if (k > MAXIT) nrerror("Series failed in ei");
				return sum+log(x)+EULER;
		} else {
				sum=0.0;
				term=1.0;
				for (k=1;k<=MAXIT;k++) {
						prev=term;
						term *= k/x;
						if (term < EPS) break;
						if (term < prev) sum += term;
						else {
								sum -= prev;
								break;
						}
				}
				return exp(x)*(1.0+sum)/x;
		}
}

double B_BK2022(double V, double Vm, double rf, int N) {
    // Our version of S&C '16's equation
    double Ne_t = 1;
    double k = Vm/V;
    double M = rf;
    double Q2;
    int T = NFACTOR*N;
    double prod = 1;
    assert(k > 0);
    assert(k < 1);
    assert(Vm > 0);
    assert(V > 0);

	  /* printf("expi(-12.1)=%g\n", gsl_sf_expint_Ei(-12.1)); */
	  /* printf("expi(12.1123)=%g\n", gsl_sf_expint_Ei(12.11123)); */
    for (int t=1; t <= T; t++) {
        Q2 = 2/M * (-((pow(-1 + exp(k*t),2)/(exp(2*k*t)*k) + 
              2*t*gsl_sf_expint_Ei(-2*k*t) - 2*t* gsl_sf_expint_Ei(-(k*t)))/(-1 + k)) + 
              ((-2*exp(k*(-2 + M)*t - M*t) * 
              pow(-1 + exp(((-(k*(-2 + M)) + M)*t)/2.),2))/ 
                (k*(-2 + M) - M) - 2*t* gsl_sf_expint_Ei(((k*(-2 + M) - M)*t)/2.) + 
              2*t* gsl_sf_expint_Ei((k*(-2 + M) - M)*t))/(-1 + k));

				double Ne = N*exp(-V/2 * Q2);
        prod = prod * (1-0.5/Ne);
	    	//printf("prod=%g, Ne=%g, Q2=%g\n", prod, Ne, Q2);
        Ne_t += prod;
    }
    return 0.5*Ne_t/N;
}




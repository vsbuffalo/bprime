#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_errno.h>

#define WARN 1

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

void check_status(int status) {
  if (!status) return;
	printf ("gsl error: %s\n", gsl_strerror(status));
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
    double a, b, c, d; // for the gsl_sf_expint_Ei_e results
	  gsl_sf_result res_a, res_b, res_c, res_d;
    int status_a, status_b, status_c, status_d;
  	gsl_set_error_handler_off();

	  /* printf("expi(-12.1)=%g\n", gsl_sf_expint_Ei(-12.1)); */
	  /* printf("expi(12.1123)=%g\n", gsl_sf_expint_Ei(12.11123)); */
    int bad = 0;
    for (int t=1; t <= T; t++) {
        status_a = gsl_sf_expint_Ei_e(-2*k*t, &res_a);
        status_b = gsl_sf_expint_Ei_e(-(k*t), &res_b);
        status_c = gsl_sf_expint_Ei_e(((k*(-2 + M) - M)*t)/2., &res_c);
        status_d = gsl_sf_expint_Ei_e((k*(-2 + M) - M)*t, &res_d);
        //check_status(status_a);
        //check_status(status_b);
        //check_status(status_c);
        //check_status(status_d);
        a = res_a.val;
        b = res_b.val;
        c = res_c.val;
        d = res_d.val;
        int error = status_a || status_b || status_c || status_d;
        if (WARN && error) {
          printf("warning: gsl error (V=%g, Vm=%g, rf=%g, N=%d)\n", V, Vm, rf, N);   
          printf("  a=%s, b=%s, c=%s, d=%s\t", gsl_strerror(status_a), 
                                               gsl_strerror(status_b), 
                                               gsl_strerror(status_c), 
                                               gsl_strerror(status_d));
 
          bad = 1;
        }
        // from tests, underflows in these values happen and they can be
        // set to zero
        if (error) {
          if (GSL_EUNDRFLW == status_c) {
            c = 0;
					} else if (GSL_EUNDRFLW == status_d) {
            d = 0;
					} else {
							printf("error: uncaught GSL error (V=%g, Vm=%g, rf=%g, N=%d)\n", V, Vm, rf, N);   
							printf("  a=%s, b=%s, c=%s, d=%s\t", gsl_strerror(status_a), 
                                                   gsl_strerror(status_b), 
                                                   gsl_strerror(status_c), 
                                                   gsl_strerror(status_d));
              exit(1);
					}
        }
        Q2 = 2/M * (-((pow(-1 + exp(k*t),2)/(exp(2*k*t)*k) + 
              2*t*a - 2*t*b )/(-1 + k)) + 
              ((-2*exp(k*(-2 + M)*t - M*t) * 
              pow(-1 + exp(((-(k*(-2 + M)) + M)*t)/2.),2))/ 
                (k*(-2 + M) - M) - 2*t*c + 
              2*t*d)/(-1 + k));

				double Ne = N*exp(-V/2 * Q2);
        prod = prod * (1-0.5/Ne);
	    	//printf("prod=%g, Ne=%g, Q2=%g\n", prod, Ne, Q2);
        Ne_t += prod;
    }
    //if (bad)
	  //  printf("Ne = %g\n", 0.5*Ne_t/N);
    return 0.5*Ne_t/N;
}




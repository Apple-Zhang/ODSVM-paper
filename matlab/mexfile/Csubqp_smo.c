/*==========================================================
 * Csubqp_smo.c
 *========================================================*/

#include <math.h>
#include "mex.h"
// #include "blas.h"

double ddot_(double *sx, double *sy, long int nn)
{
  long int i, m, iincx, iincy;
  double stemp;
  long int ix, iy;

  /* forms the dot product of two vectors.
     uses unrolled loops for increments equal to one.
     jack dongarra, linpack, 3/11/78.
     modified 12/3/93, array(1) declarations changed to array(*) */

  stemp = 0.0;
  if (nn > 0)
  {
    m = nn-4;
    for (i = 0; i < m; i += 5)
    stemp += sx[i] * sy[i] + sx[i+1] * sy[i+1] + sx[i+2] * sy[i+2] +
                sx[i+3] * sy[i+3] + sx[i+4] * sy[i+4];

    for ( ; i < nn; i++)        /* clean-up loop */
    stemp += sx[i] * sy[i];
  }

  return stemp;
} /* ddot_ */

inline double get_cy_elem(size_t yi, size_t index, double c) {
    return (index == yi) ? c : 0;
}

inline double box(double up, double low, double value) {
    if (value >= up) {
        return up;
    }
    else if (value <= low) {
        return low;
    }
    else {
        return value;
    }
}

/* The computational routine */
void smo(double *Q, double *tau, double c, size_t yi, size_t k, double *out_alpha)
{
    double EPSILON = 1e-4;
    size_t MAX_ITER = 200;

    // index number
    size_t lp = 0;
    size_t j;
    size_t r;
    size_t s;

    double lt;
    double gt;
    double p;
    double q;
    double Qrr, Qss, Qrs;
    double ar, as;
    double delta;
    double gradj;

    /* set alpha as zeros */
    for (j = 0; j < k; j++) {
        out_alpha[j] = 0;
    }

    // the first loop selected indecies
    r = yi;
    s = rand() % k;

    do
    {
        Qrr = Q[r + r*k];
        Qss = Q[s + s*k];
        Qrs = Q[r + s*k];
        ar = out_alpha[r];
        as = out_alpha[s];

        // obtain box constraint
        lt = get_cy_elem(yi, r, c) - ar;
        gt = as - get_cy_elem(yi, s, c);
        
        // qp problem
        q = Qrr + Qss - 2*Qrs;
        p = ddot_(Q + r*k, out_alpha, k) - ddot_(Q + s*k, out_alpha, k) + tau[r] - tau[s];
        // p = ar*Qrr - as*Qss + Qrs * (ar - as) + tau[r] - tau[s]; // old code.

        if (q < 1e-15) {
            if (p >= 0) {
                delta = gt;
            }
            else {
                delta = lt;
            }
        }
        else {
            delta = box(lt, gt, -p/q);
        }
        out_alpha[r] += delta;
        out_alpha[s] -= delta;

        // check optimality
        double rho_max = -2.33e12;
        double rho_min =  2.33e12;
        size_t idmx = 0;
        size_t idmn = 0;
        for (j = 0; j < k; j++) {
            // compute gradient of entry i
            gradj = ddot_(Q + j*k, out_alpha, k) + tau[j];

            if (gradj > rho_max) {
                gradj = rho_max;
                idmx = j;
            }
            if (out_alpha[j] < get_cy_elem(yi, j, c) && gradj < rho_min) {
                gradj = rho_min;
                idmn = j;
            }
        }

        if (rho_min + EPSILON >= rho_max) {
            return;
        }

        r = idmn;
        s = idmx;
        lp++;
    } while (lp < MAX_ITER);
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *Q;               /* kxk input matrix */
    double *tau;
    double c;                /* input scalar */
    size_t yi;
    size_t k;            /* size of matrix */
    double *alpha;           /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=4) {
        mexErrMsgIdAndTxt("ODSVM:SMO:nrhs","Four inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("ODSVM:SMO::nlhs","One output required.");
    }
    
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[0]) || 
         mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("ODSVM:SMO:notDouble","Input matrix Q must be type double.");
    }

    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[1]) || 
         mxIsComplex(prhs[1]) ||
        !(mxGetN(prhs[1]) == 1 || mxGetM(prhs[1]) == 1)) {
        mexErrMsgIdAndTxt("ODSVM:SMO:notDouble","Input tau must be a double vector.");
    }

    /* make sure the first input argument is scalar */
    if( !mxIsDouble(prhs[2]) || 
         mxIsComplex(prhs[2]) ||
         mxGetNumberOfElements(prhs[2])!=1 ) {
        mexErrMsgIdAndTxt("ODSVM:SMO:notScalar","Parameter c must be a scalar.");
    }

    /* make sure the first input argument is scalar */
    if(!mxIsInt32(prhs[3]) || 
         mxIsComplex(prhs[3]) ||
         mxGetNumberOfElements(prhs[3])!=1 ) {
        mexErrMsgIdAndTxt("ODSVM:SMO:notScalar","yi must be an integer.");
    }
    


    /* create a pointer to the real data in the input Q  */
    #if MX_HAS_INTERLEAVED_COMPLEX
    Q = mxGetDoubles(prhs[0]);
    #else
    Q = mxGetPr(prhs[0]);
    #endif
    
    /* get dimensions of the input matrix (nclass) */
    k = mxGetN(prhs[0]);
    
    /* create a pointer to the real data in the input tau  */
    #if MX_HAS_INTERLEAVED_COMPLEX
    tau = mxGetDoubles(prhs[1]);
    #else
    tau = mxGetPr(prhs[1]);
    #endif
    
    /* get the value of the scalar input  */
    c = mxGetScalar(prhs[2]);

    /* get yi */
    yi = mxGetScalar(prhs[3]);

    /* check bound */
    if (yi > k) {
        mexErrMsgIdAndTxt("ODSVM:SMO:OutOfBound","yi should not be larger than k.");
    }

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1, (mwSize)k, mxREAL);

    /* get a pointer to the real data in the output matrix */
    #if MX_HAS_INTERLEAVED_COMPLEX
    alpha = mxGetDoubles(plhs[0]);
    #else
    alpha = mxGetPr(plhs[0]);
    #endif

    /* call the computational routine */
    smo(Q, tau, c, yi-1, (size_t)k, alpha);
}

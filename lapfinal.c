#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void laplace_init_border ( float *in, int n, int m );

float stencil ( float v1, float v2, float v3, float v4 )
{
  return (v1 + v2 + v3 + v4) * 0.25; // Multiplying instead of dividing
}

/*
 * laplace_step
 * Now returns max error
 */
float laplace_step ( float *in, float *out, int n, int m )
{
  int i, j;
  float error=0.0f;
  // Parallelizing stencil
#pragma omp parallel for reduction(max:error) collapse(2)
  for ( j=1; j < n-1; j++ ) {
    for ( i=1; i < m-1; i++ ) {
      int idx = j*m+i; // Re-use precomputed idx : avoids multiplications
      const float l = in[idx-1];
      const float r = in[idx+1];
      const float u = in[idx-m];
      const float d = in[idx+m];
      out[idx]= stencil(l,r,d,u);
      error = fmaxf( error, fabsf( out[idx] - in[idx] ));
    }
  }
  return sqrtf(error);
}

/*
float laplace_error ( float *old, float *new, int n, int m )
{
  int i, j;
  float error=0.0f;
  // Parallelizing error computation : reduction
#pragma omp parallel for reduction(max:error) collapse(2)
  for ( j=1; j < n-1; j++ )
    for ( i=1; i < m-1; i++ )
      error = fmaxf( error, fabsf( old[j*m+i] - new[j*m+i] ));

  return sqrtf(error);
}
*/


void laplace_init ( float *in, int n, int m )
{
  memset(in, 0, n*m*sizeof(float));
  laplace_init_border(in, n, m);
}

void laplace_init_border ( float *in, int n, int m )
{
  // Split this border init
  // We will use it for both initial matrix & secondary matrix
  int i, j;
  const float pi  = 2.0f * asinf(1.0f);
  for (i=0; i<m; i++)  in[    i    ] = 0.f;
  for (i=0; i<m; i++)  in[(n-1)*m+i] = 0.f;
  for (j=0; j<n; j++)  in[   j*m   ] = sinf(pi*j / (n-1));
  for (j=0; j<n; j++)  in[ j*m+m-1 ] = sinf(pi*j / (n-1))*expf(-pi);
}

int main(int argc, char** argv)
{
  int n = 4096, m = 4096;
  int iter_max = 100;
  float *A, *Anew,*Atmp; // Atmp to help matrix pointer swap
    
  const float tol = 1.0e-5f;
  float error= 1.0f;    

  // get runtime arguments: n, m and iter_max 
  if (argc>1) {  n        = atoi(argv[1]); }
  if (argc>2) {  m        = atoi(argv[2]); }
  if (argc>3) {  iter_max = atoi(argv[3]); }

  A    = (float*) malloc( n*m*sizeof(float) );
  Anew = (float*) malloc( n*m*sizeof(float) );

  //  set boundary conditions
  laplace_init (A, n, m);
  laplace_init_border(Anew, n, m);

  A[(n/128)*m+m/128] = 1.0f; // set singular point

  printf("Jacobi relaxation Calculation: %d x %d mesh,"
         " maximum of %d iterations\n", 
         n, m, iter_max );

  int iter = 0;

  while ( error > tol && iter < iter_max )
  {
    iter++;
    error = laplace_step (A, Anew, n, m);
    //error= laplace_error (A, Anew, n, m);
    Atmp = Anew; // 
    Anew = A;    // Swapping both matrixes instead of copying one to the other
    A = Atmp;    //
    if (iter % (iter_max/10) == 0) printf("%5d, %0.6f\n", iter, error);
  }
  printf("Total Iterations: %5d, ERROR: %0.6f, ", iter, error);
  printf("A[%d][%d]= %0.6f\n", n/128, m/128, A[(n/128)*m+m/128]);

  free(A); 
  free(Anew);
}

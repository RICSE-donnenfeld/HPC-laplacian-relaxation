// lap2.c — Optimized OpenMP Jacobi (Laplace 2D)
// Applies: buffer swapping, loop fusion, loop interchange, strength reduction, code motion.

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static inline float stencil4(float r, float l, float u, float d) {
  // strength reduction: multiply by 0.25f instead of divide by 4
  return (r + l + u + d) * 0.25f;
}

static void laplace_init(float *A, int n, int m) {
  // code motion: one memset for interior + borders, then set boundaries
  memset(A, 0, (size_t)n * m * sizeof(float));

  const float pi = 2.0f * asinf(1.0f);
  const float e_minus_pi = expf(-pi);

  // loop fusion for left/right borders; keep i inner in data loops elsewhere
  for (int j = 0; j < n; ++j) {
    float s = sinf(pi * j / (n - 1));
    A[j * m + 0]     = s;                // left boundary
    A[j * m + m - 1] = s * e_minus_pi;   // right boundary
  }
  // top/bottom already zero by memset, left as documentation
  // A[0, :] and A[n-1, :] = 0
}

int main(int argc, char **argv) {
  int n = 4096, m = 4096;
  int iter_max = 100;
  if (argc > 1) n        = atoi(argv[1]);
  if (argc > 2) m        = atoi(argv[2]);
  if (argc > 3) iter_max = atoi(argv[3]);

  float *A    = (float*) malloc((size_t)n * m * sizeof(float));
  float *Anew = (float*) malloc((size_t)n * m * sizeof(float));
  if (!A || !Anew) {
    fprintf(stderr, "Allocation failed\n"); free(A); free(Anew); return 1;
  }

  laplace_init(A, n, m);
  // singular checkpoint (clamp to interior if tiny grids)
  int sj = (n/128 > 0 ? n/128 : 1);
  int si = (m/128 > 0 ? m/128 : 1);
  if (sj >= n-1) sj = n-2;
  if (si >= m-1) si = m-2;
  A[sj * m + si] = 1.0f;

  // initialize Anew = A to avoid first-iteration reads of garbage
  memcpy(Anew, A, (size_t)n * m * sizeof(float));

  const float tol = 1.0e-5f;
  const int report = (iter_max > 0) ? ((iter_max/10) ? (iter_max/10) : 1) : 1;

  // timing (optional)
  double t0;
  #pragma omp parallel
  {
    #pragma omp master
    {
      printf("Jacobi relaxation: %d x %d, iter_max=%d, tol=%g, threads=%d\n",
             n, m, iter_max, tol, omp_get_num_threads());
      t0 = omp_get_wtime();
    }
  }

  float error = INFINITY;
  int iter = 0;

  // strength reduction: compare max |diff| and take sqrt once per iter
  // loop fusion: compute Anew AND track max diff in the same loop
  while (error > tol && iter < iter_max) {
    float maxAbsDiff = 0.0f;

    #pragma omp parallel for collapse(2) reduction(max:maxAbsDiff) schedule(static)
    for (int j = 1; j < n - 1; ++j) {
      for (int i = 1; i < m - 1; ++i) {
        const int idx = j * m + i;          // code motion: reuse idx
        const float r = A[idx + 1];
        const float l = A[idx - 1];
        const float u = A[idx - m];
        const float d = A[idx + m];
        const float newv = stencil4(r, l, u, d);
        Anew[idx] = newv;

        const float diff = fabsf(newv - A[idx]);
        if (diff > maxAbsDiff) maxAbsDiff = diff;
      }
    }

    // buffer swapping: avoid copying interior back
    float *tmp = A; A = Anew; Anew = tmp;

    error = sqrtf(maxAbsDiff);  // strength reduction: one sqrt per iteration
    ++iter;

    if (iter % report == 0)
      printf("%5d, %0.6f\n", iter, error);
  }

  double t1 = omp_get_wtime();
  printf("Total Iterations: %d, ERROR: %.6f, A[%d][%d]= %.6f, time=%.3f s\n",
         iter, error, sj, si, A[sj * m + si], t1 - t0);

  free(A);
  free(Anew);
  return 0;
}

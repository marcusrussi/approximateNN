#include "ann.h"
#include "randNorm.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>
#include "gpu_comp.h"

void genRand(size_t n, size_t d, ftype *points) {
  for(size_t i = 0; i < n * d; i++)
    points[i] = rand_norm();
}

double compute_score(size_t n, size_t k, size_t d,
		     const ftype *points, const size_t *guess, double *scb,
		     double *scc);
double compute_score_query(size_t n, size_t k, size_t d, size_t ycnt,
			   const ftype *points, const ftype *y,
			   const size_t *guess, double *scb,
			   double *scc);

int main(int argc, char **argv) {
  size_t n = 1000, k = 10, d = 80, tries = 10, average_over = 100;
  size_t ycnt = 50, rb = 6, rlenb = 1, ra = 1, rlena = 1;
  char progress = 0, use_cpu = 0, use_y = 0;
  opterr = 0;
  int c;
  srandom(time(NULL));
  while((c = getopt(argc, argv, "n:k:d:t:o:y:b:s:a:r:hvcz")) != -1)
    switch(c) {
    case '?':
      fprintf(stderr, "Unknown option %c or missing argument.\n", optopt);
    case 'h':
      fprintf(stderr, "Legal options are:\n"
	      "\t-a n\t\tSet the post-Walsh rotation count to n (default 1)\n"
	      "\t-b n\t\tSet the pre-Walsh rotation count to n (default 6)\n"
	      "\t-c\t\tUse the CPU instead of the GPU\n"
	      "\t-d n\t\tSet the dimensionality to n (default 80)\n"
	      "\t-h\t\tPrint this help text\n"
	      "\t-k n\t\tRequest n nearest neighbors (default 10)\n"
	      "\t-n n\t\tSet the point count to n (default 1000)\n"
	      "\t-o n\t\tSet the number of repetitions to average over to n"
	      " (default 100)\n"
	      "\t-r n\t\tSet the post-Walsh rotation size to n (default 1)\n"
	      "\t-s n\t\tSet the pre-Walsh rotation size to n (default 1)\n"
	      "\t-t n\t\tSet the try-count to n (default 10)\n"
	      "\t-v\t\tIncrease verbosity\n"
	      "\t-y n\t\tSet the count of query points to n\n"
	      "\t-z\t\tSame as -y 50\n");
      exit(0);
    case 'n':
      n = strtol(optarg, NULL, 0);
      break;
    case 'k':
      k = strtol(optarg, NULL, 0);
      break;
    case 'd':
      d = strtol(optarg, NULL, 0);
      break;
    case 't':
      tries = strtol(optarg, NULL, 0);
      break;
    case 'o':
      average_over = strtol(optarg, NULL, 0);
      break;
    case 'y':
      ycnt = strtol(optarg, NULL, 0);
    case 'z':
      use_y = 1;
      break;
    case 'b':
      rb = strtol(optarg, NULL, 0);
      break;
    case 's':
      rlenb = strtol(optarg, NULL, 0);
      break;
    case 'a':
      ra = strtol(optarg, NULL, 0);
      break;
    case 'r':
      rlena = strtol(optarg, NULL, 0);
      break;
    case 'v':
      progress = 1;
      break;
    case 'c':
      use_cpu = 1;
      break;
    default:
      fprintf(stderr, "Can\'t happen!\n");
      exit(1);
    }
  if(!use_cpu)
    gpu_init();
  double score = 0, scb = 0, scc = 0;
  ftype *points = malloc(sizeof(ftype) * n * d);
  if(use_y) {
    save_t save;
    genRand(n, d, points);
    free(precomp(n, k, d, points, tries, rb, rlenb, ra, rlena, &save, NULL,
		 use_cpu));
    if(progress)
      printf("Precomputation finished.\n");
    ftype *y = malloc(sizeof(ftype) * ycnt * d);
    for(size_t i = 0; i < average_over; i++) {
      size_t *stuff;
      genRand(ycnt, d, y);
      stuff = query(&save, points, ycnt, y, NULL, use_cpu);
      score += compute_score_query(n, k, d, ycnt, points, y, stuff,
				   &scb, &scc);
      free(stuff);
      if(progress)
	printf("%zu ", i + 1), fflush(stdout);
    }
    free(y);
    free_save(&save);
  } else
    for(size_t i = 0; i < average_over; i++) {
      size_t *stuff;
      genRand(n, d, points);
      stuff = precomp(n, k, d, points, tries, rb, rlenb, ra, rlena,
		      NULL, NULL, use_cpu);
      score += compute_score(n, k, d, points, stuff, &scb, &scc);
      free(stuff);
      if(progress)
	printf("%zu ", i + 1), fflush(stdout);
    }
  if(!use_cpu)
    gpu_cleanup();
  free(points);
  if(progress)
    putchar('\n');
  printf("Average index score for %s (on %cPU): %g.\nProb correct: %g.\n"
	 "Max index score: %g\n",
	 use_y? "query" :  "comp",
	 use_cpu? 'C' : 'G',
	 (score / average_over - k * (k - 1) / 2) / k,
	 1 - scb / average_over,
	 scc / average_over / k);
}

// guess is y by k, ansinv is y by n.
double cscore(size_t y, size_t n, size_t k,
	      const size_t *guess, const size_t *ansinv, double *foo,
	      double *bar);

typedef struct {
  size_t point;
  ftype dist;
} pairedup;

int cpoint(const void *p, const void *q) {
  const pairedup *a = p, *b = q;
  if(a->dist > b->dist)
    return(1);
  else if(a->dist < b->dist)
    return(-1);
  else
    return(0);
}

// inv_ans(n, 0, ans)[i * n + i] == ULONG_MAX.
// inv_ans(n, 0, ans)[i * n + j] == inv_ans(n, n - 1, ans)[i * (n - 1) + j].
size_t *inv_ans(size_t y, size_t n, const pairedup *ans);
void compdists(size_t ycnt, size_t n, size_t d,
	       pairedup *p, const ftype *y, const ftype *points);

double compute_score(size_t n, size_t k, size_t d,
		     const ftype *points, const size_t *guess, double *scb,
		     double *scc) {
  pairedup *ans = malloc(sizeof(pairedup) * n * (n - 1));
  for(size_t i = 0; i < n; i++)
    for(size_t j = 0, k = 0; j < n - 1; j++, k++) {
      if(k == i)
	k++;
      ans[i * (n - 1) + j].point = k;
    }
  compdists(n, n - 1, d, ans, points, points);
  for(size_t i = 0; i < n; i++)
    qsort(ans + i * (n - 1), n - 1, sizeof(pairedup), cpoint);
  size_t *ia = inv_ans(n, 0, ans);
  free(ans);
  double f = cscore(n, n, k, guess, ia, scb, scc);
  free(ia);
  return(f);
}

double compute_score_query(size_t n, size_t k, size_t d, size_t ycnt,
			   const ftype *points, const ftype *y,
			   const size_t *guess, double *scb,
			   double *scc) {
  pairedup *ans = malloc(sizeof(pairedup) * ycnt * n);
  for(size_t i = 0; i < ycnt; i++)
    for(size_t j = 0; j < n; j++)
      ans[i * n + j].point = j;
  compdists(ycnt, n, d, ans, y, points);
  for(size_t i = 0; i < ycnt; i++)
    qsort(ans + i * n, n, sizeof(pairedup), cpoint);
  size_t *ia = inv_ans(ycnt, n, ans);
  free(ans);
  double f = cscore(ycnt, n, k, guess, ia, scb, scc);
  free(ia);
  return(f);
}

void compdists(size_t ycnt, size_t n, size_t d,
	       pairedup *p, const ftype *y, const ftype *points) {
  ftype *sds = malloc(sizeof(ftype) * d);
  for(size_t i = 0; i < ycnt; i++)
    for(size_t j = 0; j < n; j++) {
      pairedup *q = p + i * n + j;
      size_t qp = q->point;
      for(size_t k = 0; k < d; k++) {
	double f = y[i * d + k] - points[qp * d + k];
	sds[k] = f * f;
      }
      for(size_t step = d; step / 2; step /= 2) {
	if(step % 2)
	  sds[0] += sds[step / 2] + sds[step - 1];
	for(size_t z = step % 1; z < step / 2; z++)
	  sds[z] += sds[z + step / 2];
      }
      q->dist = sds[0];
    }
  free(sds);
}

size_t *inv_ans(size_t y, size_t n, const pairedup *ans) {
  char selfsame = !n;
  size_t q, *a;
  if(selfsame)
    n = (q = y) - 1;
  else
    q = n;
  a = malloc(sizeof(size_t) * q * y);
  for(size_t i = 0; i < y; i++) {
    for(size_t j = 0; j < n; j++)
      a[i * q + ans[i * n + j].point] = j;
    if(selfsame)
      a[i * q + i] = ULONG_MAX;
  }
  return(a);
}

double cscore(size_t y, size_t n, size_t k,
	      const size_t *guess, const size_t *ansinv, double *d,
	      double *foo) {
  double f = 0, g = 0;
  size_t max = 0;
  for(size_t i = 0; i < y; i++)
    for(size_t j = 0; j < k; j++) {
      size_t l = ansinv[i * n + guess[i * k + j]];
      f += l;
      g += l >= k;
      if(l > max)
	max = l;
    }
  *d += g / y / k;
  *foo += max;
  return(f / y);
}


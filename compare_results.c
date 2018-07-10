#include "ann.h"
#include "randNorm.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include "gpu_comp.h"

void genRand(size_t n, size_t d, double *points) {
  for(size_t i = 0; i < n * d; i++)
    points[i] = rand_norm();
}

// Note: 1024 ulp = 1 diff on ints ('twould be any diff on doubles,
// but GPU/CPU differences)
double cdiff_save(save_t *a, save_t *b);

size_t diffcount(size_t n, size_t k, size_t *p, size_t *q) {
  size_t c = 0;
  for(size_t i = 0; i < n * k; i++)
    c += p[i] != q[i];
  return(c);
}

int main(int argc, char **argv) {
  size_t n = 1000, k = 10, d = 80, tries = 10, average_over = 100;
  size_t ycnt = 50, rb = 6, rlenb = 1, ra = 1, rlena = 1;
  char progress = 0, use_y = 0;
  opterr = 0;
  int c;
  while((c = getopt(argc, argv, "n:k:d:t:o:y:b:s:a:r:hvz")) != -1)
    switch(c) {
    case '?':
      fprintf(stderr, "Unknown option %c or missing argument.\n", optopt);
    case 'h':
      fprintf(stderr, "Legal options are:\n"
	      "\t-a n\t\tSet the post-Walsh rotation count to n (default 1)\n"
	      "\t-b n\t\tSet the pre-Walsh rotation count to n (default 6)\n"
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
    default:
      fprintf(stderr, "Can\'t happen!\n");
      exit(1);
    }
  gpu_init();
  double score = 0;
  double *points = malloc(sizeof(double) * n * d);
  FILE *randomf = fopen("/dev/urandom", "r");
  if(use_y) {
    save_t save;
    genRand(n, d, points);
    precomp(n, k, d, points, tries, rb, rlenb, ra, rlena, &save, 0);
    if(progress)
      printf("Precomputation finished.\n");
    double *y = malloc(sizeof(double) * ycnt * d);
    for(size_t i = 0; i < average_over; i++) {
      size_t *stuff, *other;
      char foo[256], bar[256];
      genRand(ycnt, d, y);
      fread(foo, 1, 256, randomf);
      memcpy(bar, foo, 256);
      setstate(foo);
      stuff = query(&save, points, ycnt, y, 0);
      setstate(bar);
      other = query(&save, points, ycnt, y, 1);
      score += diffcount(ycnt, k, stuff, other);
      free(stuff);
      free(other);
      if(progress)
	printf("%zu ", i + 1), fflush(stdout);
    }
    free(y);
    free_save(&save);
  } else
    for(size_t i = 0; i < average_over; i++) {
      save_t stuff, other;
      genRand(n, d, points);
      precomp(n, k, d, points, tries, rb, rlenb, ra, rlena,
	      &stuff, 0);
      precomp(n, k, d, points, tries, rb, rlenb, ra, rlena,
	      &other, 1);
      score += cdiff_save(&stuff, &other);
      free_save(&stuff);
      free_save(&other);
      if(progress)
	printf("%zu ", i + 1), fflush(stdout);
    }
  gpu_cleanup();
  free(points);
  if(progress)
    putchar('\n');
  printf("Average diffs for %s: %g\n",
	 ycnt? "query" :  "comp",
	 score / average_over);
}

double cdiff_save(save_t *a, save_t *b) {
  if(a->tries != b->tries || a->d_short != b->d_short ||
     a->k != b->k || a->d_long != b->d_long || a->n != b->n)
    return(ULONG_MAX);
  double c = 0;
  for(size_t i = 0; i < a->n * a->k; i++)
    c += a->graph[i] != b->graph[i];
  for(size_t i = 0; i < a->tries * a->d_short * a->d_long; i++)
    c += labs(((long *)a->bases)[i] - ((long *)b->bases)[i]) / 1024.;
  for(int i = 0; i < a->tries; i++) {
    c += labs(((long *)a->row_means)[i] - ((long *)b->row_means)[i]) / 1024.;
    if(a->par_maxes[i] != b->par_maxes[i])
      return(ULONG_MAX);
    for(size_t j = 0; j < a->par_maxes[i] << a->d_short; j++)
      c += a->which_par[i][j] != b->which_par[i][j];
  }
  return(c);
}

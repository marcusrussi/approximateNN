#include "timing.h"
#include "ann.h"
#include "randNorm.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

void genRand(size_t n, size_t d, ftype *points) {
  for(size_t i = 0; i < n * d; i++)
    points[i] = rand_norm();
}

int main(int argc, char **argv) {
  size_t n = 1000, k = 10, d = 80, tries = 10, average_over = 100;
  size_t ycnt = 0, rb = 6, rlenb = 1, ra = 1, rlena = 1;
  char save_test = 0, progress = 0;
  opterr = 0;
  int c;
  srandom(time(NULL));
  while((c = getopt(argc, argv, "n:k:d:t:o:y:b:s:a:r:hzv")) != -1)
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
	      "\t-z\t\tTurn on saving for queries\n");
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
    case 'z':
      save_test = 1;
      break;
    case 'v':
      progress = 1;
      break;
    default:
      fprintf(stderr, "Can\'t happen!\n");
      exit(1);
    }
  double time_used = 0;
  ftype *points = malloc(sizeof(ftype) * n * d);
  if(ycnt) {
    save_t save;
    genRand(n, d, points);
    precomp(n, k, d, points, tries, rb, rlenb, ra, rlena, &save, 1);
    if(progress)
      printf("Precomputation finished.\n");
    ftype *y = malloc(sizeof(ftype) * ycnt * d);
    for(size_t i = 0; i < average_over; i++) {
      size_t *stuff;
      tval start, end;
      genRand(ycnt, d, y);
      gettm(start);
      stuff = query(&save, points, ycnt, y, 1);
      gettm(end);
      free(stuff);
      time_used += td(start, end);
      if(progress)
	printf("%zu ", i + 1), fflush(stdout);
    }
    free(y);
    free_save(&save);
  } else
    for(size_t i = 0; i < average_over; i++) {
      save_t save;
      tval start, end;
      size_t *stuff;
      genRand(n, d, points);
      gettm(start);
      stuff = precomp(n, k, d, points, tries, rb, rlenb, ra, rlena,
		      save_test? &save : NULL, 1);
      gettm(end);
      if(save_test)
	free_save(&save);
      else
	free(stuff);
      time_used += td(start, end);
      if(progress)
	printf("%zu ", i + 1), fflush(stdout);
    }
  free(points);
  if(progress)
    putchar('\n');
  printf("Average time for %s (on CPU): %gs\n",
	 ycnt? "query" : save_test? "comp (with save)" : "comp (no save)",
	 time_used / average_over / 1000000000);
}

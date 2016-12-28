#pragma once

#ifdef __linux__
#include <sys/types.h>
#include <inttypes.h>
#endif

#include "opt.h"
#include <math.h>
#include <errno.h>
#include <time.h>
#include <assert.h>
#include <syslog.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <cfloat>
#include <stdio.h>
#include <vector>
#include "linmath.h"

#define CHIMERA_OK 0

struct range_t {
  float min, max;
};

static float random_gauss(float mu, float sigma)
{
	const float epsilon = FLT_EPSILON;
	const float two_pi = 2.0 * M_PI;

	static float z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	float u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

static inline float randomf()
{
  const int m = 2048;
  const float s = (float)m;
  return (random() % m) / s;
}

static inline float randomf(range_t& range)
{
  float s = randomf();
  return (s * (range.max - range.min)) + range.min;
}

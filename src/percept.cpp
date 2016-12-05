#include "percept.hpp"

float randomf(range_t& range)
{
  float s = (float)(1 << 20);
  return (rand() * s * range.max / s) + range.min;
}

void Percept::permute()
{
  for(int i = parameter_ranges.size(); i--;)
  {
    parameters[i] = randomf(parameter_ranges[i]);
  }
}

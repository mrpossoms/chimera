#include "percept.hpp"

void Percept::permute()
{
  for(int i = parameter_ranges.size(); i--;)
  {
    parameters[i] = randomf(parameter_ranges[i]);
  }
}

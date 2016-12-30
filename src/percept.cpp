#include "percept.hpp"

void Percept::add_parameter(range_t range)
{
  parameter_ranges.push_back(range);
}

void Percept::permute()
{
  if(!parameters)
  {
    parameters = (float*)calloc(parameter_ranges.size(), sizeof(float));
  }

  for(int i = parameter_ranges.size(); i--;)
  {
    parameters[i] = randomf(parameter_ranges[i]);
  }
}

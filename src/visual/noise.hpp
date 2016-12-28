#pragma once

#include "base.hpp"
#include "material.hpp"

class UniformNoise : public Material {
public:
  UniformNoise(int width, int height, range_t red, range_t green, range_t blue);
  ~UniformNoise();

  void sample_at(unsigned int x, unsigned int y, void* textel);
  void permute();

  range_t* noise_params;
};

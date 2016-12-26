#include "noise.hpp"

union noise_params_t {
  struct{
    range_t red, green, blue;
  };
  range_t v[3];
};

UniformNoise::UniformNoise(
  int width, int height,
  range_t red, range_t green, range_t blue) : Material(width, height)
{
  rand_range[0] = red;
  rand_range[1] = green;
  rand_range[2] = blue;

  noise_params_t* p = new noise_params_t();
  p->red   = red;
  p->green = green;
  p->blue  = blue;

  parameters = (float*)p;
}

UniformNoise::~UniformNoise()
{
  delete parameters;

  // Material::~Material();
}

void UniformNoise::permute()
{
  noise_params_t* p = (noise_params_t*)parameters;

  // enforce range constraints
  for(int i = 3; i--;)
  {
    p->v[i].min = randomf(rand_range[i]);
    p->v[i].max = randomf(rand_range[i]);

    if(p->v[i].min > p->v[i].max)
    {
      float temp = p->v[i].min;
      p->v[i].min = p->v[i].max;
      p->v[i].max = temp;
    }
  }

  Material::permute();
}

void UniformNoise::sample_at(unsigned int x, unsigned int y, void* textel)
{
  noise_params_t* p = (noise_params_t*)parameters;
  rgb_t* t = (rgb_t*)textel;

  for(int i = 3; i--;)
  {
    t->v[i] = p->v[i].max * 255.f;//(uint8_t)(randomf(p->v[i]) * 255.f);
  }
}

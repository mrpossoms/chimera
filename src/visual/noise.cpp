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
  add_parameter(red);
  add_parameter(green);
  add_parameter(blue);

  noise_params_t* p = new noise_params_t();
  p->red   = red;
  p->green = green;
  p->blue  = blue;

  parameters = (float*)p;
  noise_params = (range_t*)parameters;
}

UniformNoise::~UniformNoise()
{
  delete parameters;

  // Material::~Material();
}

void UniformNoise::permute()
{
  Material::permute();

  // enforce range constraints
  for(int i = 3; i--;)
  {
    noise_params[i].min = randomf(parameter_ranges[i]);
    noise_params[i].max = randomf(parameter_ranges[i]);

    if(noise_params[i].min > noise_params[i].max)
    {
      float temp = noise_params[i].min;
      noise_params[i].min = noise_params[i].max;
      noise_params[i].max = temp;
    }
  }
}

void UniformNoise::sample_at(unsigned int x, unsigned int y, void* textel)
{
  rgb_t* t = (rgb_t*)textel;

  for(int i = 3; i--;)
  {
    t->v[i] = (uint8_t)(randomf(noise_params[i]) * 255.f);
  }
}

#pragma once

#include "base.hpp"
#include "percept.hpp"

union rgb_t {
  struct {
    uint8_t r, g, b;
  };
  uint8_t v[3];
};

class Material : public Percept {
public:
  Material(unsigned int width, unsigned int height);
  ~Material();

  unsigned int get_width();
  unsigned int get_height();

  // must be implemented by user to generate texture data
  virtual void sample_at(unsigned int x, unsigned int y, void* textel) = 0;

  // rendering api dependent
  void use(int id);

  void permute();

protected:
  void*  tag; // implementation dependent
  rgb_t*  data;
  unsigned int width, height;
};

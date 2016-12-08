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
  Material(int width, int height);
  ~Material();

  unsigned int get_width();
  unsigned int get_height();

  // must be implemented by user to generate texture data
  virtual void sample_at(unsigned int x, unsigned int y, void* textel) = 0;

  void permute();
  void render();

  rgb_t* data;

protected:
  void*  tag; // implementation dependent
  unsigned int width, height;
};

#pragma once

class Scene {
public:
  virtual int  tag() = 0;

  virtual void permute() = 0;
  virtual void render() = 0;
  virtual int  save(const char* path) = 0;
};

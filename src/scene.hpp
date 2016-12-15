#pragma once

#define SAMPLE_MAGIC 0x26c1f8b9;
#define LABEL_MAGIC 0xfd1802a2;

typedef struct {
  uint32_t magic_number;
  uint32_t sample_size;
} samples_hdr_t;

typedef samples_hdr_t labels_hdr_t;

class Scene {
public:
  virtual int  tag() = 0;

  virtual void permute() = 0;
  virtual void render() = 0;
  virtual int  save(const char* path) = 0;
};

#ifndef _VISUAL_H
#define _VISUAL_H

#include "noise.hpp"
#include "mesh.h"
#include "material.hpp"
#include "viewer.hpp"
#include <png.h>

struct chimera_visual_options {
  bool write_blob;
  bool is_rgb;
  int do_trunc;
};

extern chimera_visual_options VIS_OPTS;

static void write_png_file_grey(
  const char* path,
  int width,
  int height,
  const void* image){

  int y;
  FILE *fp = fopen(path, "wb");
  const uint8_t* buffer = (const uint8_t*)image;

  if(!fp) abort();

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) abort();

  png_infop info = png_create_info_struct(png);
  if (!info) abort();

  if (setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(
    png,
    info,
    width, height,
    8,
    PNG_COLOR_TYPE_GRAY,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info(png, info);

  png_bytep rows[height];
  for(int i = height; i--;)
  {
    rows[i] = (png_bytep)(buffer + i * width);
  }

  png_write_image(png, rows);
  png_write_end(png, NULL);

  fclose(fp);
}

static void write_png_file_rgb(
  const char* path,
  int width,
  int height,
  const void* image){

  int y;
  FILE *fp = fopen(path, "wb");
  const rgb_t* buffer = (const rgb_t*)image;

  if(!fp) abort();

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png) abort();

  png_infop info = png_create_info_struct(png);
  if (!info) abort();

  if (setjmp(png_jmpbuf(png))) abort();

  png_init_io(png, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(
    png,
    info,
    width, height,
    8,
    PNG_COLOR_TYPE_RGB,
    PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT,
    PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info(png, info);

  png_bytep rows[height];
  for(int i = height; i--;)
  {
    rows[i] = (png_bytep)(buffer + i * width);
  }

  png_write_image(png, rows);
  png_write_end(png, NULL);

  fclose(fp);
}

static void append_blob_file(int fd, const void* buf, size_t bytes, uint32_t tag)
{
  assert(write(fd, buf, bytes) == bytes);
  assert(write(fd, &tag, sizeof(uint32_t)) == sizeof(uint32_t));
}

#endif

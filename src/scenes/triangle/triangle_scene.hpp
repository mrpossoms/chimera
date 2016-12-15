#pragma once

#include <GLFW/glfw3.h> // TODO move to move general visual scene
#include "chimera.h"
#include "visual/visual.h"

#include "triangle_mesh.hpp"

#define SAMPLE_WIDTH 128
#define SAMPLE_HEIGHT 128

void write_png_file_grey(
  const char* path,
  int width,
  int height,
  const uint8_t* buffer){
  int y;

  FILE *fp = fopen(path, "wb");
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

void write_png_file_rgb(
  const char* path,
  int width,
  int height,
  const rgb_t* buffer){
  int y;

  FILE *fp = fopen(path, "wb");
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


class TriangleScene : public Scene {
public:
  TriangleScene()
  {
    // glfw setup
    if(!glfwInit())
    {
      syslog(LOG_ERR, "Failed to initalize GLFW3");
      exit(-1);
    }

    glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);
    win = glfwCreateWindow(SAMPLE_WIDTH >> 1, SAMPLE_HEIGHT >> 1, "Chimera", NULL, NULL);
    if(!win)
    {
      syslog(LOG_ERR, "Failed to open window");
      exit(-2);
    }

    glfwMakeContextCurrent(win);

    glEnable(GL_TEXTURE_2D);

    // glPixelTransferf(GL_RED_SCALE, 0.3086);
    // glPixelTransferf(GL_GREEN_SCALE, 0.6094);
    // glPixelTransferf(GL_BLUE_SCALE, 0.0820);

    // Chimera scene setup
    view = new Viewer(SAMPLE_WIDTH >> 1, SAMPLE_HEIGHT >> 1);
    view->view.look = Vec3(0, 0, 1);

    range_t r = { 0, 1 };
    tri_noise = new UniformNoise(SAMPLE_WIDTH >> 1, SAMPLE_HEIGHT >> 1, r, r, r);
    bg_noise = new UniformNoise(SAMPLE_WIDTH, SAMPLE_HEIGHT, r, r, r);
  }

  ~TriangleScene()
  {
    glfwTerminate();
  }

  int tag()
  {
    return view->in_view(tri.bounding_sphere) ? 1 : 0;
  }

  void permute()
  {
    tri.permute();
    tri_noise->permute();
    bg_noise->permute();
  }

  void render()
  {
    glClear(GL_COLOR_BUFFER_BIT);

    char title[256];
    sprintf(title, "Chimera %d", tag());
    glfwSetWindowTitle(win, title);
    bg_noise->render();
    glDrawPixels(
      bg_noise->get_width(),
      bg_noise->get_height(),
      GL_RGB, GL_UNSIGNED_BYTE,
      (void*)bg_noise->data
    );

    view->render();
    tri_noise->render();
    tri.render();
    glFinish();
    glfwPollEvents();
  }

  int save(const char* path)
  {
    rgb_t frame_buffer[(view->width * 2) * (view->height * 2)];

    glReadPixels(
      0, 0,
      view->width * 2, view->height * 2,
      GL_RGB, GL_UNSIGNED_BYTE,
      (void*)frame_buffer
    );

    char file_path[256];
    sprintf(file_path, "%s/%lX%lX-%d.png", path, time(NULL), random(), tag());
    write_png_file_rgb(file_path, view->width * 2, view->height * 2, frame_buffer);

    return CHIMERA_OK;
  }

private:
  GLFWwindow* win;
  TriangleMesh tri;
  UniformNoise *tri_noise, *bg_noise;
  Viewer* view;
};

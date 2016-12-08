#pragma once

#include <GLFW/glfw3.h> // TODO move to move general visual scene
#include "chimera.h"
#include "visual/visual.h"

#include "triangle_mesh.hpp"


void write_png_file(
  const char* path,
  int width,
  int height,
  const rgb_t* buffer)
  {
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

    win = glfwCreateWindow(256, 256, "Chimera", NULL, NULL);
    if(!win)
    {
      syslog(LOG_ERR, "Failed to open window");
      exit(-2);
    }

    glfwMakeContextCurrent(win);

    glEnable(GL_TEXTURE_2D);

    // Chimera scene setup
    view = new Viewer(256, 256);
    view->view.look = Vec3(0, 0, 1);

    range_t r = { 0, 1 };
    noise = new UniformNoise(256, 256, r, r, r);
  }

  ~TriangleScene()
  {
    glfwTerminate();
  }

  int tag()
  {
    return 0;
  }

  void permute()
  {
    tri.permute();
    noise->permute();
    mat4x4_translate(tri.transform, 0, 0, 10);
  }

  void render()
  {
    glClear(GL_COLOR_BUFFER_BIT);

    view->render();
    noise->render();
    tri.render();

    glfwSwapBuffers(win);
    glfwPollEvents();
    sleep(1);
  }

  int save(const char* path)
  {
    rgb_t frame_buffer[view->width * view->height * 4];

    glReadPixels(
      0, 0,
      view->width * 2, view->height * 2,
      GL_RGB, GL_UNSIGNED_BYTE,
      (void*)frame_buffer
    );

    char file_path[256];
    sprintf(file_path, "%s/%lX%lX-%d.png", path, time(NULL), random(), tag());
    write_png_file(file_path, view->width * 2, view->height * 2, frame_buffer);

    return CHIMERA_OK;
  }

private:
  GLFWwindow* win;
  TriangleMesh tri;
  UniformNoise* noise;
  Viewer* view;
};

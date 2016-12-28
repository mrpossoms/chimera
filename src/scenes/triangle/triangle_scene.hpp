#pragma once

#ifdef __APPLE__
#include <GLFW/glfw3.h> // TODO move to move general visual scene
#elif __linux__
#include <GL/freeglut.h>
#endif


#include "chimera.h"
#include "visual/visual.h"

#include "triangle_mesh.hpp"
#include "poly_mesh.hpp"

#define SAMPLE_WIDTH 112
#define SAMPLE_HEIGHT 112

void write_png_file_grey(
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

void write_png_file_rgb(
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

void append_blob_file(int fd, const void* buf, size_t bytes, uint32_t tag)
{
  assert(write(fd, buf, bytes) == bytes);
  assert(write(fd, &tag, sizeof(uint32_t)) == sizeof(uint32_t));
}

class TriangleScene : public Scene {
public:
  TriangleScene()
  {

#ifdef __APPLE__
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
    syslog(LOG_INFO, "GLFW window created");
#elif __linux__
    setenv ("DISPLAY", ":0", 0);
    glutInit(&ARGC, (char**)ARGV);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(SAMPLE_WIDTH, SAMPLE_WIDTH);
    glutCreateWindow("Chimera");
#endif

    glEnable(GL_TEXTURE_2D);
    // glEnable(GL_LIGHTING);
    // glEnable(GL_LIGHT0);

    // Chimera scene setup
    glViewport(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
    view = new Viewer(SAMPLE_WIDTH, SAMPLE_HEIGHT);
    view->view.look = Vec3(0, 0, 1);

    range_t one = { 0, 1 };
    range_t low = { 0.4, 0.6 };
    tri_noise = new UniformNoise(32, 32, one, one, one);
    bg_noise = new UniformNoise(SAMPLE_WIDTH, SAMPLE_HEIGHT, one, one, one);

    if(VIS_OPTS.write_blob)
    {
      BLOB_FD = open("data/training_blob", O_CREAT | O_WRONLY | VIS_OPTS.do_trunc, 0666);
      assert(BLOB_FD >= 0);
    }
  }

  ~TriangleScene()
  {
    if(VIS_OPTS.write_blob)
    {
      close(BLOB_FD);
    }

#ifdef __APPLE__
    glfwTerminate();
#endif
  }

  int tag()
  {
    return in_view;
  }

  void permute()
  {
    in_view = random() % 2;
  }

  void render()
  {
    glClear(GL_COLOR_BUFFER_BIT);

    tri.permute();

    bg_noise->permute();
    bg_noise->render();
    glDrawPixels(
      bg_noise->get_width(),
      bg_noise->get_height(),
      GL_RGB, GL_UNSIGNED_BYTE,
      (void*)bg_noise->data
    );
    view->render();

    tri_noise->permute();
    for(int i = 4; i--;)
    {
      tri_noise->render();
      bg_poly.permute();
      bg_poly.render();
    }

    if(in_view)
    {
      for(int i = 3; i--;)
      {
        float mean = tri_noise->noise_params[i].max - tri_noise->noise_params[i].min;

        if(tri_noise->noise_params[i].min > 0.5)
        {
          tri_noise->noise_params[i].max = tri_noise->noise_params[i].min;
          tri_noise->noise_params[i].min = 0;
        }
        else
        {
          tri_noise->noise_params[i].min = tri_noise->noise_params[i].max;
          tri_noise->noise_params[i].max = 1;
        }
      }

      tri_noise->permute();
      tri_noise->render();
      tri.render();
    }

    glFinish();

#ifdef __APPLE__
    glfwPollEvents();
#endif
  }

  int save(const char* path)
  {

    unsigned int pixels = (view->width) * (view->height);
    uint8_t grey_buffer[pixels];
    rgb_t color_buffer[pixels];

    const void* frame_buffer;
    size_t buffer_size;
    void (*png_encoder)(const char*, int, int, const void*);

    glReadPixels(
      0, 0,
      view->width, view->height,
      GL_RGB, GL_UNSIGNED_BYTE,
      (void*)color_buffer
    );

    if(VIS_OPTS.is_rgb)
    {
      png_encoder = write_png_file_rgb;
      buffer_size = sizeof(color_buffer);
      frame_buffer = (const void*)color_buffer;
    }
    else
    {
      // convert to grey scale
      for(int i = pixels; i--;){
        rgb_t color = color_buffer[i];
        grey_buffer[i] = color.r / 3 + color.g / 3 + color.b / 3;
      }

      // compute false variance to check for data validity
      float mu = grey_buffer[random() % pixels] / 255.f, var = 0;
      for(int i = pixels; i--;)
      {
        var += powf(mu - grey_buffer[i] / 255.f, 2);
      }

      printf("variance %f\n", var);

      png_encoder = write_png_file_grey;
      buffer_size = sizeof(grey_buffer);
      frame_buffer = (const void*)grey_buffer;
    }

    if(VIS_OPTS.write_blob)
    {
      append_blob_file(BLOB_FD, frame_buffer, buffer_size, tag());
    }
    else
    {
      char file_path[256];
      sprintf(file_path, "%s/%lX%lX-%d.png", path, time(NULL), random(), tag());
      png_encoder(file_path, view->width, view->height, frame_buffer);
    }

    return CHIMERA_OK;
  }

private:
#ifdef __APPLE__
  GLFWwindow* win;
#endif
  TriangleMesh tri;
  PolyMesh bg_poly;
  UniformNoise *tri_noise, *bg_noise;
  Viewer* view;
  bool in_view;
  int BLOB_FD;
};

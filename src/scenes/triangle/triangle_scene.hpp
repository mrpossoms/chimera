#pragma once

#ifdef __APPLE__
#include <GLFW/glfw3.h> // TODO move to move general visual scene
#elif __linux__
#include <GL/freeglut.h>
#endif


#include "chimera.h"
#include "visual/visual.h"

#include "triangle_mesh.hpp"

#define SAMPLE_WIDTH 128
#define SAMPLE_HEIGHT 128

#define GEN_GRAY

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

void append_blob_file(int fd, const void* buf, size_t bytes, uint32_t tag)
{
  write(fd, buf, bytes);
  write(fd, &tag, sizeof(uint32_t));
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

    // glPixelTransferf(GL_RED_SCALE, 0.3086);
    // glPixelTransferf(GL_GREEN_SCALE, 0.6094);
    // glPixelTransferf(GL_BLUE_SCALE, 0.0820);

    // Chimera scene setup
    glViewport(0, 0, SAMPLE_WIDTH, SAMPLE_HEIGHT);
    view = new Viewer(SAMPLE_WIDTH, SAMPLE_HEIGHT);
    view->view.look = Vec3(0, 0, 1);



    range_t l = { 0, .5 }, u = { 0.5, 0.1 }, z = { 0, 0 };
    tri_noise = new UniformNoise(SAMPLE_WIDTH >> 1, SAMPLE_HEIGHT >> 1, z, u, z);
    bg_noise = new UniformNoise(SAMPLE_WIDTH, SAMPLE_HEIGHT, z, z, z);

#ifdef GEN_GRAY
    BLOB_FD = open("data/training_blob", O_CREAT | O_WRONLY | O_TRUNC, 0666);
    assert(BLOB_FD >= 0);
#endif
  }

  ~TriangleScene()
  {
#ifdef GEN_GRAY
    close(BLOB_FD);
#endif

#ifdef __APPLE__
    glfwTerminate();
#endif
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

#ifdef __APPLE__
    glfwPollEvents();
#endif
  }

  int save(const char* path)
  {

    unsigned int pixels = (view->width) * (view->height);
#ifdef GEN_GRAY
    uint8_t grey_buffer[pixels];
#endif

    rgb_t frame_buffer[pixels];

    glReadPixels(
      0, 0,
      view->width, view->height,
      GL_RGB, GL_UNSIGNED_BYTE,
      (void*)frame_buffer
    );

#ifdef GEN_GRAY
    // convert to grey scale
    for(int i = pixels; i--;){
      rgb_t color = frame_buffer[i];
      grey_buffer[i] = color.r / 3 + color.g / 3 + color.b / 3;
    }
    // write_png_file_grey(file_path, view->width * 2, view->height * 2, grey_buffer);
    append_blob_file(BLOB_FD, grey_buffer, sizeof(grey_buffer), tag());
#else
    char file_path[256];
    sprintf(file_path, "%s/%lX%lX-%d.png", path, time(NULL), random(), tag());
    write_png_file_rgb(file_path, view->width, view->height, frame_buffer);
    //append_blob_file(BLOB_FD, frame_buffer, sizeof(frame_buffer), tag());
#endif

    return CHIMERA_OK;
  }

private:
#ifdef __APPLE__
  GLFWwindow* win;
#endif
  TriangleMesh tri;
  UniformNoise *tri_noise, *bg_noise;
  Viewer* view;
  int BLOB_FD;
};

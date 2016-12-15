#include "material.hpp"
#include "glbase.hpp"

struct gl_material_t {
  GLuint id;
  bool generated;
};

#define GL_PROXY gl_material_t* gl = (gl_material_t*)tag;

Material::Material(int width, int height)
{
  this->width = width; this->height = height;
  data = new rgb_t[width * height];

  // allocate a new gl_material_t object to track
  // material properties from within GL
  gl_material_t* gl = new gl_material_t();
  tag = (void*)gl;

  glGenTextures(1, &gl->id);
  gl->generated = false;
}

Material::~Material()
{
  GL_PROXY

  glDeleteTextures(1, &gl->id);
  delete gl;
  delete data;
}

unsigned int Material::get_width()
{
  return width;
}

unsigned int Material::get_height()
{
  return height;
}

void Material::render()
{
  GL_PROXY

  glBindTexture(GL_TEXTURE_2D, gl->id);

  if(!gl->generated)
  {
    // iterate over each row
    for(int y = height; y--;)
    {
      rgb_t* row = data + y * width;

      // fill each column textel
      for(int x = width; x--;)
      {
        //row[x].r = 128;
        //row[x].g = row[x].b = 0;
        sample_at(x, y, row + x);
      }
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_RGB,
      width,
      height,
      0,
      GL_RGB,
      GL_UNSIGNED_BYTE,
      (void*)data
    );
    gl->generated = true;
  }
}

void Material::permute()
{
  GL_PROXY
  Percept::permute();
  gl->generated = false;
}

#pragma once

#include "chimera.h"
#include "visual/visual.h"

class TriangleMesh : public Mesh {
public:
  TriangleMesh()
  {
    // NOOP
  }

  const void* vertex_buffer()
  {
    return (const void*)vertices.data();
  }

  void permute()
  {
    float theta = randomf() * M_PI * 2;
    float texcoord_scale = powf(randomf(), 3);

    vertices.clear();

    for(int i = 3; i--;)
    {
      const float t = (M_PI * 2) / 3.f;
      vertex_t v = {
        .a = Vec3(cos(i * t + theta), sin(i * t + theta), 0),
        .b = Vec3(cos(i * t) / 2.f + 0.5f, sin(i * t) / 2.f + 0.5f, 0),
      };

      v.b *= texcoord_scale;

      vertices.push_back(v);
    }

    generated = false;
  }

  void render()
  {
    // if(!generated)
    // {
    //   glGenBuffers(1, &id);
    //
    //   generated = true;
    // }
    glPushMatrix();
    glMultMatrixf((const GLfloat*)transform);

    glBegin(GL_TRIANGLES);

    for(int i = 3; i--;)
    {
      // glColor3f(1, 0, 0);
      glTexCoord2f(vertices[i].b.x, vertices[i].b.y);
      glVertex2f(vertices[i].a.x, vertices[i].a.y);
    }

    glEnd();
  }
private:
  bool generated;
  GLuint id;
};

#pragma once

#include "chimera.h"
#include "visual/visual.h"

class NgonMesh : public Mesh {
public:
  GLenum render_style = GL_TRIANGLE_FAN;

  NgonMesh(int min, int max, range_t yaw, range_t pitch)
  {
    // NOOP
    range_t x = { .min = -1, .max = 1 };
    range_t y = { .min = -1, .max = 1 };
    range_t z = { .min = 2, .max = 5 };
    range_t vert_count = { .min = (float)min, .max = (float)max };
    range_t line_thickness = { 0, 3 };
    add_parameter(x);
    add_parameter(y);
    add_parameter(z);
    add_parameter(vert_count);
    add_parameter(yaw);
    add_parameter(pitch);
    add_parameter(line_thickness);
  }

  const void* vertex_buffer()
  {
    return (const void*)vertices.data();
  }

  void permute(float w)
  {
    Percept::permute();
    float theta = randomf() * M_PI * 2;
    float texcoord_scale = 1;//powf(randomf(), 2);

    vertices.clear();
    int verts = roundf(randomf(parameter_ranges[3]));
    float t = (M_PI * 2) / verts;

    for(int i = verts; i--;)
    {
      vertex_t v = {
        .a = Vec3(cos(i * t + theta), sin(i * t + theta), 0),
        .b = Vec3(cos(i * t) / 2.f + 0.5f, sin(i * t) / 2.f + 0.5f, 0),
      };

      v.b *= texcoord_scale;

      vertices.push_back(v);
    }

    // printf("%f - %f %f - %f\n", parameter_ranges[0].min, parameter_ranges[0].max, parameter_ranges[1].min, parameter_ranges[1].max);

     Vec3 p(
      randomf(parameter_ranges[0]),
      randomf(parameter_ranges[1]),
      randomf(parameter_ranges[2])
    );

    float s = p.z;
    mat4x4 inv_proj = {
      { s, 0, 0, 0 },
      { 0, s, 0, 0 },
      { 0, 0, 1, 0 },
      { 0, 0, 0, 1 },
    };

    mat4x4_mul_vec3(bounding_sphere.origin.v, inv_proj, p.v);

    mat4x4 temp;
    mat4x4_identity(transform);
    mat4x4_translate_in_place(transform,
      bounding_sphere.origin.x,
      bounding_sphere.origin.y,
      bounding_sphere.origin.z
    );
    mat4x4_rotate_Y(temp, transform, randomf(parameter_ranges[4]));
    mat4x4_rotate_X(transform, temp, randomf(parameter_ranges[5]));


    generated = false;
  }

  float in_view(Viewer& viewer)
  {
    float frac = 0;
    Vec3 vec;

    for(int i = vertices.size(); i--;)
    {
      mat4x4_mul_vec3(vec.v, transform, vertices[i].a.v);
      if(viewer.in_view(vec))
      {
        frac += 1 / (float)vertices.size();
      }
    }

    return frac;
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

    glBegin(render_style);
    for(int i = vertices.size(); i--;)
    {
      const range_t r = { 0.25, 1 };
      glColor3f(randomf(r), randomf(r), randomf(r));
      glTexCoord2f(vertices[i].b.x, vertices[i].b.y);
      glVertex2f(vertices[i].a.x, vertices[i].a.y);
    }
    glEnd();

    glPopMatrix();
  }
private:
  bool generated;
  GLuint id;
};

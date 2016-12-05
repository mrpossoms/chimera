#include "viewer.hpp"

void Viewer::render()
{
  mat4x4 MV, P;

  Vec3 up = VEC3_UP;
  Vec3 left;

  vec3_mul_cross(left.v, up.v, view.look.v);
  vec3_mul_cross(up.v, left.v, view.look.v);

  mat4x4_perspective(P, view.fov, height / width, 0.1, 100);
  mat4x4_look_at(MV, view.position.v, view.look.v, up.v);

  glMatrixMode(GL_PROJECTION);
  glLoadMatrixf((float*)P);

  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixf((float*)MV);
}

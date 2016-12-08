#include "task.hpp"

Task::Task(Scene* scene, task_ctx_t ctx)
{
  target = ctx;

  current.tag_count = ctx.tag_count;
  current.tag_distribution = new unsigned int[ctx.tag_count]();
  current.samples = 0;

  for(int i = ctx.tag_count; i--;)
  {
    current.tag_distribution[i] = 0;
  }

  this->scene = scene;
}

Task::~Task()
{
  delete current.tag_distribution;
}

void Task::run()
{
  while(current.samples < target.samples)
  {
    scene->permute();

    int tag = scene->tag();
    if(current.tag_distribution[tag] < target.tag_distribution[tag])
    {
      // if we don't already have too many of that tag
      // then we will keep it
      current.tag_distribution[tag]++;
      current.samples++;

      scene->render();
      assert(scene->save(target.data_path) == CHIMERA_OK);
    }
    else
    {
      printf("%d@ %d > %d\n", tag, current.tag_distribution[tag], target.tag_distribution[tag]);
      continue;
    }
  }
}

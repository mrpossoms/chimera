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
  int percent_complete = 0, last_percent_complete = 0;
  time_t then = time(NULL);

  syslog(LOG_INFO, "Generating training data.\n");

  while(current.samples < target.samples)
  {
    scene->permute();

    percent_complete = current.samples * 100 / target.samples;

    if(percent_complete > last_percent_complete)
    {
      time_t now = time(NULL);
      float minutes_remaining = (now - then) * (100 - percent_complete) / 60.f;
      then = now;

      syslog(LOG_INFO, "%d%% - %f min remaining\n", percent_complete, minutes_remaining);

      last_percent_complete = percent_complete;
    }

    int tag = scene->tag();

    // find the next desired tag to keep training samples uniform
    unsigned int wanted_tag = 0;
    unsigned int number = current.tag_distribution[0];
    for(int i = current.tag_count; i--;)
    {
      if(number > current.tag_distribution[i])
      {
        wanted_tag = i;
        number = current.tag_distribution[i];
      }
    }

    if(tag == wanted_tag)
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
      continue;
    }
  }
}

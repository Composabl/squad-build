# SkillTeacher Hyperparameters

Teacher training knobs are provided through `Skill(...)` kwargs.

```python
skill = Skill(
    "my-skill",
    MyTeacher,
    training_cycles=100,
    train_batch_size=4000,
    workers=1,
    envs_per_worker=1,
    learner_workers=0,
    num_cpus_per_worker=1,
    num_gpus_per_worker=0,
    num_gpus_per_learner_worker=0,
    num_cpus_per_learner_worker=1,
    fc_layers=[256, 256],
)
```

## Commonly tuned fields

- `training_cycles`: PPO iterations per outer cycle for this skill.
- `train_batch_size`: rollout samples per PPO update.
- `workers` and `envs_per_worker`: parallel data collection.
- `fc_layers`: hidden layer sizes.

## Important distinction

- `training_cycles` (on `Skill`) != `train_cycles` (passed to `trainer.train(...)`).
- Total optimization work is roughly the product of both.

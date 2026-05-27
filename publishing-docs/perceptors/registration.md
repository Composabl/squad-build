# Perceptor Registration

Register perceptors on the agent, then reference their output keys in skill sensor filters.

```python
agent.add_perceptor(Perceptor("kinematics", MyPerceptor))
```

Skills can then include perceptor output keys in `filtered_sensor_space()`.

## Stateful behavior

Perceptor instance state is not automatically reset between episodes unless your implementation does so explicitly.

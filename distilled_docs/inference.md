### Accessing Trained Models

After training completes, model checkpoints are saved to:

```
{output_dir}/{skill_name}/policy_checkpoint_{timestamp}/
```

Example:

```bash
ls benchmarks/control/policy_checkpoint_20260423_101542/
```

To load a policy for inference:

```python
from ray.rllib.algorithms.ppo import PPO

# Load the checkpoint
policy = PPO.from_checkpoint("benchmarks/control/policy_checkpoint_20260423_101542/")
action = policy.compute_single_action(observation)[0]
```

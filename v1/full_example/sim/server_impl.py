from amesa_core.networking.sim.server_amesa import ServerAmesa

from v1.full_example.sim.sim import GreenhouseSim


class SimImpl(ServerAmesa):
    def __init__(self, *args, **kwargs):
        self.env_init = kwargs.get("env_init", {})
        self.env = GreenhouseSim(self.env_init)

    async def make(self, env_id: str, env_init: dict):
        return self.env.make(env_id, env_init)

    async def sensor_space_info(self):
        return self.env.sensor_space_info()

    async def action_space_info(self):
        return self.env.action_space_info()

    async def action_space_sample(self):
        return self.env.action_space_sample()

    async def reset(self):
        return self.env.reset()

    async def step(self, action):
        return self.env.step(action)

    async def close(self):
        self.env.close()

    async def set_scenario(self, scenario):
        self.env.set_scenario(scenario)

    async def get_scenario(self):
        return self.env.get_scenario()

    async def get_render(self):
        return self.env.get_render()

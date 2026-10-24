"""The OpenDrawer task (Isaac Lab): open the top drawer, put the charger in, close the drawer.

Copied from RoboticArmSim (``tasks/manager_based/opendrawer``) unchanged except the charger asset's path, which now
points into this package. Importing this module registers ``OpenDrawer-ManagerBased-v0``; it needs a running Isaac
Sim app (``vig_dagger.isaac.launch``) because the config imports Isaac Lab.
"""

import gymnasium as gym

gym.register(
    id="OpenDrawer-ManagerBased-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.open_drawer_env_cfg:OpenDrawerEnvCfg"},
)

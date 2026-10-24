"""OpenDrawer constants shared by both environments (no simulator imports, so training code can use them too)."""

ENV_ID = "OpenDrawer-ManagerBased-v0"
TASK_NAME, SLUG = "open drawer", "open_drawer"  # dataset task name and folder
POLICY_CAMERAS = ("left", "wrist")  # the policy's image inputs; demo files store them as obs/rgb_camera/<camera>_cam
MAX_EPISODE_STEPS = 1200  # demos, DAgger and evaluation
POLICY, EXPERT = 0, 1  # who drove a DAgger step (the ``controller`` arrays)

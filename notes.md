## Getting DMC working
.bashrc on Arva
export MUJOCO_GL="glfw"
export MJLIB_PATH=$HOME/.mujoco/mujoco200_linux/bin/libmujoco200.so
export MJKEY_PATH=$HOME/.mujoco/mujoco200_linux/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200_linux/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200_linux/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200_linux/mjkey.txt
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.2.0

.bashrc on Skynet
export MUJOCO_GL="egl"

command:
Need to run this before running dmc2gym code
xvfb-run -a -s "-screen 0 1400x900x24" bash

Ensure This Runs:
python -c "import mujoco_py"

Might Need:
sudo apt-get install libglew-dev
pip install patchelf

## Problem 1
GLFWError: (65542) b'GLX: GLX version 1.3 is required'
  warnings.warn(message, GLFWError)
CRITICAL:absl:GLEW initalization error: Missing GL version
CRITICAL:absl:OpenGL version 1.5 or higher required
CRITICAL:absl:OpenGL ARB_framebuffer_object required
CRITICAL:absl:OpenGL ARB_vertex_buffer_object required

Solution:

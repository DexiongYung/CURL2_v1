## Arva
.bashrc
export MUJOCO_GL="glfw"
export MJLIB_PATH=$HOME/.mujoco/mujoco200_linux/bin/libmujoco200.so
export MJKEY_PATH=$HOME/.mujoco/mujoco200_linux/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200_linux/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200_linux/
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200_linux/mjkey.txt
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.2.0


## Skynet
.bashrc
export MUJOCO_GL="egl"

need mujoco200 unzipped under .mujoco folder in $HOME as well as key. Install ONLY dmc2gym


## Sanity Check
Ensure This Runs:
python -c "import mujoco_py"

Might Need:
sudo apt-get install libglew-dev
pip install patchelf


##Problem 1
Get the following error when running train.py
_glfwPlatformGetTls: Assertion `tls->posix.allocated == 1' failed.


Solution:
xvfb-run -a -s "-screen 0 1400x900x24" bash


## Problem 2
GLFWError: (65542) b'GLX: GLX version 1.3 is required'
  warnings.warn(message, GLFWError)
CRITICAL:absl:GLEW initalization error: Missing GL version
CRITICAL:absl:OpenGL version 1.5 or higher required
CRITICAL:absl:OpenGL ARB_framebuffer_object required
CRITICAL:absl:OpenGL ARB_vertex_buffer_object required

Solution:

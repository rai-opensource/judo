## The GUI
The GUI is built using `viser <https://viser.studio/main/>`_. `judo` automatically creates sliders, buttons, and other GUI elements for you when the corresponding configs are registered.

### Sliders
Sliders represent float or int parameters. We provide the `slider` decorator to make it easy to adjust the min, max, and step size of the corresponding slider GUI element. For example:
```python
@slider("my_param1", min_val, max_val, step_size)
@dataclass
def MyTaskConfig(TaskConfig):
    my_param1: float = 1.0
    my_param2: int = 2  # this will have a default slider
```
If you don't supply the decorator, the slider limits are automatically registered as
```python
min_val = 0
max_val = default * 2
step_size = (max_val - min_val) / 100
```

In the GUI, if you type a value in the text box next to a slider, it will adjust it live (warning: this means that for values of more than one character, it reads the updates **as the characters are typed**). If the value typed in the field exceeds the limits, the limits update to accommodate.

### `numpy` Array Fields
For `numpy` array fields, we provide the convenience field utility `np_1d_field` for fine-grained control over the sliders for all of the array elements. Additionally, sometimes the array field might correspond to some visualization you want to show in the GUI. For example, in the `cylinder_push` task, we have a field that specifies the goal position. The `np_1d_field` function also provides ways to show a small spherical visualization. In the case of the `cylinder_push` task, the config looks like this:
```python
from judo.utils.fields import np_1d_field

@dataclass
class CylinderPushConfig(TaskConfig):
    """Reward configuration for the cylinder push task."""

    w_pusher_proximity: float = 0.5
    w_pusher_velocity: float = 0.0
    w_cart_position: float = 0.1
    pusher_goal_offset: float = 0.25
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0]),
        names=["x", "y"],
        mins=[-1.0, -1.0],
        maxs=[1.0, 1.0],
        steps=[0.01, 0.01],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
    )
```
The `vis_name` field specifies the name of the visualization object in the GUI backend, so it doesn't matter that much. `xyz_vis_indices` indicates which indices of the array to use for the x, y, and z coordinates of the visualization. If you set it to `None`, it will use the default values specified in `xyz_vis_defaults`.

<p align="center">
  <img src="/judo/_static/images/cylinder_push.gif" alt="cylinder_push_config" width="640">
</p>

### Visualizing Traces
In the above gif, you can see visualizations of the traces of the sim rollouts from the controller. We can automatically register traces for visualization by editing the corresponding XML of the `mujoco` system.

In particular, we must have a `framepos` sensor with the substring `trace` in its name. For example, in the `cylinder_push` task, we have the following:
```xml
<!-- We strip out everything but the relevant sites and sensors in this XML. -->
<mujoco model="cylinder_push">
    <body name="pusher" pos="0 0 0">
      <site pos="0 0 0.15" name="pusher_site"/>
    </body>
    <body name="cart" pos="0 0 0">
      <site pos="0 0 0.15" name="cart_site"/>
    </body>
  </worldbody>

  <!-- These sensors are what create the traces in the GUI! -->
  <sensor>
    <framepos name="trace_pusher" objtype="site" objname="pusher_site"/>
    <framepos name="trace_cart" objtype="site" objname="cart_site"/>
  </sensor>
</mujoco>
```

### Link Sharing
`viser` provides support for sharing a URL of your current GUI to send to others, who can then interact with `judo` remotely! This is done by clicking the sharing icon in the top right of the GUI menu to request a URL that can be copy and pasted to others.

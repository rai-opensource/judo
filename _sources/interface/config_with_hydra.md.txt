## Configuration with `hydra`
`judo` is mainly built on top of `dora <https://dora-rs.ai/>`_ for interprocess communication and `hydra <https://hydra.cc/docs/intro/>`_ for configuration management.

These two elements work together in tandem - the dataflow specification for `dora` specifies things like the expected inputs/outputs of each node, the sources (aka "topics" in ROS2 language), etc. The `hydra` yaml file includes the dataflow spec while also specifying parameters required to construct the nodes. See `judo/configs/judo_default` for an example on usage.

You can write custom configs for things like overrides, instantiating custom nodes, or setting initial tasks/optimizers. Once a configuration file has been written, usage is simple:
```bash
judo -cp /absolute/path/to/config/folder -cn <config_name>
```
where `config_name` corresponds to the name of a yaml file. Concretely, the above will try to start the `judo` stack using the config located at `/absolute/path/to/config/folder/config_name.yaml`. If the `-cp` flag is ommitted, it searches for `/path/to/judo/configs/config_name.yaml`.

Note that if you supply the `-cp` flag, the path to the config folder **must be absolute**. To interpolate the absolute path, you can use the `realpath` command in the CLI. For example, in the repo root, you can run:
```bash
judo -cp $(realpath .)/example_configs -cn example
```
This will launch the `judo` stack as before, but using a custom config located in a different config folder than the one installed with `judo`! In general, you can point this anywhere, including your own projects located elsewhere. Note that if you want to use a custom config with all the default setup we provide, you must add this to the top of your config:
```yaml
defaults:
  - judo
```

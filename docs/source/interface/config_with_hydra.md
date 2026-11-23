## Configuration with `hydra`
`judo` is mainly built on top of its own `unlocked` shared-memory dataflow runtime (``judo/unlocked``) for interprocess communication and `hydra <https://hydra.cc/docs/intro/>`_ for configuration management.

These two elements work together in tandem - the graph of nodes (simulation, controller, visualization) and their connections is defined in ``judo/app/graph.py``, and the `hydra` yaml file specifies the parameters passed to that graph builder (task, optimizer, custom registrations, overrides). See `judo/configs/judo_default.yaml` for an example on usage. To inspect or edit the graph in the browser, run `judo-graph-ui`.

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

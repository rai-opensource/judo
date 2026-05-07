# FAQ and Troubleshooting

- **Connection Rejected Error.** If you get the error `[websockets.server][INFO] - connection rejected (200 OK)` when trying to open the web app, try blocking cookies for `http://localhost` in your browser. Sometimes this also doesn't work - opening in the localhost link in a private browsing window usually works. You may need to close and re-open a private session if it's getting stuck.
- **Inherited config not overriding default fields.** If you have inherited one config from another and the GUI is not reflecting default field value differences, then you have most likely forgotten the `@dataclass` decorator.
- **Meshes not showing up.** If a mesh isn't showing up in the GUI, make sure that the geoms are named. If there are multiple mesh geoms that are not named, they will not show up in the GUI.

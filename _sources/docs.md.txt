# Docs
For developers, to build docs locally, run the following in your environment from the repo root. Note that asset paths will be broken locally that work correctly on Github Pages.
```bash
# using conda
pip install -e .[docs]  # dev also includes docs

# using pixi
pixi shell -e docs  # dev also includes docs

# building the docs (both conda and pixi)
sphinx-build docs/source docs/build -b dirhtml
python -m http.server --directory docs/build 8000
```

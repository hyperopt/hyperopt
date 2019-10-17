# Hyperopt documentation

To build and run Hyperopt documentation locally, you should install `mkdocs` first,
then auto-generate files from templates, build your `mkdocs` site and then run it:

```bash
pip install mkdocs
python autogen.py
mkdocs build && mkdocs serve
```

To deploy a new version of docs run:
```bash
mkdocs gh-deploy
```

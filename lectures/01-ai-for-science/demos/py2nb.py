"""Convert a `# %%` script into a Jupyter notebook: python py2nb.py demo1_pinn_oscillator.py"""
import re
import sys

import nbformat


def convert(path):
    text = open(path).read()
    cells = []
    for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            md = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = path.replace(".py", ".ipynb")
    nbformat.write(nb, out)
    print("wrote", out)


for p in sys.argv[1:]:
    convert(p)

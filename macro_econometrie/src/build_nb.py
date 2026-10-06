"""Turn a '# %%' percent-format script into an executed .ipynb.

Usage: python src/build_nb.py nb_src/01_x.py   ->  01_x.ipynb (executed, outputs kept)
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def build(src: Path, execute: bool = True) -> Path:
    cells, cur, kind = [], [], None

    def flush():
        if kind is None or not "".join(cur).strip():
            return
        text = "".join(cur).strip("\n")
        if kind == "md":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n")]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in src.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("# %% [markdown]"):
            flush()
            cur, kind = [], "md"
        elif line.startswith("# %%"):
            flush()
            cur, kind = [], "code"
        else:
            cur.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = src.parent.parent / (src.stem + ".ipynb")
    if execute:
        NotebookClient(nb, timeout=7200, kernel_name="python3",
                       resources={"metadata": {"path": str(out.parent)}}).execute()
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = tidy(c.outputs)
    nbformat.write(nb, out)
    return out


def tidy(outputs: list) -> list:
    """Drop warning noise captured from stderr and merge consecutive chunks of the same stream."""
    kept = []
    for o in outputs:
        if o.output_type == "stream" and o.name == "stderr" and "Warning" in o.text:
            continue
        if kept and o.output_type == "stream" and kept[-1].output_type == "stream" and kept[-1].name == o.name:
            kept[-1].text += o.text
        else:
            kept.append(o)
    return kept


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print("built", build(Path(a)))

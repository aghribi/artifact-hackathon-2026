"""Render the deck's formulae to SVG with LaTeX.

HTML sub/sup/&radic; does not hold together at auditorium size, so the handful of
real formulae in talk.html are LaTeX, compiled to standalone PDFs and converted to
SVG. They are text-as-paths, so they need no fonts at the far end and scale cleanly.

    python3 -I scripts/make_math_svgs.py

Needs a LaTeX engine (tectonic or latexmk/pdflatex) and pdftocairo (poppler).
Output: math/<name>.svg, sized by the `height` the slide gives the <img>.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
TALK = HERE.parent
OUT = TALK / "math"

PREAMBLE = r"""
\documentclass[border=2pt,varwidth]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath}
\usepackage{sansmath}
\sansmath
\begin{document}
"""

# name -> body. Keep these in sync with the slides that embed them.
FORMULAE = {
    # "5. The focus control set": how each magnet is scored across the SVD modes
    "magnet_score": r"$\displaystyle \mathrm{score}_j=\sqrt{\sum_i s_i^2\,V_{ij}^2}$",
    # the response the quadrupole SVD is taken of
    "size_response": r"$\displaystyle \frac{\partial \ln \sigma}{\partial k_1}$",
    # "6. Tweak the tune, two": the planner's anchored prediction
    "anchored": (
        r"$\displaystyle \hat{y}(\delta)=y_{\text{measured}}"
        r"+\bigl[\,\mathrm{twin}(k+\delta)-\mathrm{twin}(k)\bigr]$"
    ),
}


def engine():
    for name, args in (
        ("tectonic", ["-X", "compile", "--keep-logs=false"]),
        ("latexmk", ["-pdf", "-interaction=nonstopmode"]),
        ("pdflatex", ["-interaction=nonstopmode"]),
    ):
        path = shutil.which(name) or (
            "/tmp/tectonic_dl/tectonic" if name == "tectonic"
            and pathlib.Path("/tmp/tectonic_dl/tectonic").exists() else None
        )
        if path:
            return path, args
    raise SystemExit("no LaTeX engine found (tectonic, latexmk or pdflatex)")


def main() -> int:
    binary, args = engine()
    if not shutil.which("pdftocairo"):
        raise SystemExit("pdftocairo not found (install poppler-utils)")

    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TALK, prefix=".math_build_") as workdir:
        work = pathlib.Path(workdir)
        for name, body in FORMULAE.items():
            tex = work / f"{name}.tex"
            tex.write_text(PREAMBLE + body + "\n\\end{document}\n")
            subprocess.run([binary, *args, tex.name], cwd=work, check=True, capture_output=True)
            pdf = work / f"{name}.pdf"
            if not pdf.exists():
                raise SystemExit(f"{name}: no PDF produced")
            subprocess.run(
                ["pdftocairo", "-svg", str(pdf), str(OUT / f"{name}.svg")],
                check=True,
                capture_output=True,
            )
            print(f"{name}.svg  {(OUT / f'{name}.svg').stat().st_size / 1024:.1f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

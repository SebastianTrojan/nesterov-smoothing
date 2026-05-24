# Nesterov Smoothing in Python

This repository contains a focused Python implementation of the method from:

Y. Nesterov, "Smooth minimization of non-smooth functions," *Mathematical Programming* 103(1), 127-152, 2005.

The codebase has been trimmed to the paper-aligned pieces only:

- the entropy-smoothed max-affine solver
- the simplex geometry used in the paper
- the Section 6 matrix-game reproduction

Removed from the project:

- `L∞` regression experiments
- adaptive / changing-`mu` continuation variants
- extra non-paper experiment scripts

## Project layout

```text
.
|-- nesterov_smoothing.py
|-- nesterov_smoothing_lib/
|-- reproduce_nesterov_2005.py
|-- requirements.txt
|-- test_nesterov_smoothing.py
`-- README.md
```

## Setup

### PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### CMD

```cmd
py -3 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you prefer not to activate the environment, you can run the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
```

## Running tests

```powershell
python -m unittest -v
```

## Quick examples

Run the small demo from the main module:

```powershell
python nesterov_smoothing.py
```

Run one Section 6 matrix-game instance:

```powershell
python reproduce_nesterov_2005.py --eps 0.01 --m-values 100 --n-values 100
```

Run the full Section 6 grid:

```powershell
python reproduce_nesterov_2005.py
```

Save the Section 6 results to CSV:

```powershell
python reproduce_nesterov_2005.py --csv results/section6.csv
```

## Notes

- The matrix-game solver uses the paper's fixed smoothing choice `mu = epsilon / (2 log m)`.
- The default experiment grid can take a long time on larger instances.
- `nesterov_smoothing.py` is a convenience shim over the `nesterov_smoothing_lib` package.
- `unittest` is part of the Python standard library, so only `numpy` is listed in `requirements.txt`.

## Reference

- Paper: https://webdoc.sub.gwdg.de/ebook/serien/e/CORE/dp2003-12.pdf

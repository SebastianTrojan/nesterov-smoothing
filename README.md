# Nesterov Smoothing in Python

This repository contains a Python implementation of the smoothing and accelerated minimization ideas from:

Y. Nesterov, "Smooth minimization of non-smooth functions," *Mathematical Programming* 103(1), 127-152, 2005.

The project currently includes:

- A generic finite-max smoothing solver in [nesterov_smoothing.py](nesterov_smoothing.py)
- A paper-specific matrix-game experiment runner in [reproduce_nesterov_2005.py](reproduce_nesterov_2005.py)
- Unit tests in [test_nesterov_smoothing.py](test_nesterov_smoothing.py)

## Project layout

```text
.
|-- .venv/
|-- nesterov_smoothing.py
|-- reproduce_nesterov_2005.py
|-- requirements.txt
|-- test_nesterov_smoothing.py
`-- README.md
```

## Setup

### PowerShell

Create and activate the virtual environment:

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

Run the demo from the main solver module:

```powershell
python nesterov_smoothing.py
```

Run a small matrix-game reproduction:

```powershell
python reproduce_nesterov_2005.py --eps 0.01 --m-values 100 --n-values 100
```

Run the full Section 6 grid:

```powershell
python reproduce_nesterov_2005.py
```

Save the results to CSV:

```powershell
python reproduce_nesterov_2005.py --csv results/section6.csv
```

## Notes

- The default experiment grid can take a long time. On this machine, the full run is on the order of hours rather than minutes.
- The implementation focuses on the finite-max structure and the matrix-game experiments described in the paper.
- `unittest` is part of the Python standard library, so only `numpy` is listed in `requirements.txt`.

## Reference

- Paper: https://webdoc.sub.gwdg.de/ebook/serien/e/CORE/dp2003-12.pdf

# Nesterov Smoothing Examples

This is a small Python project built around three examples from:

Y. Nesterov, "Smooth minimization of non-smooth functions," *Mathematical Programming* 103(1), 127-152, 2005.

The code stays flat at the project root. It is not a package, but the logic is now split into a few small modules:

- shared core: [nesterov_core.py](nesterov_core.py)
- matrix game: [nesterov_matrix_game.py](nesterov_matrix_game.py)
- continuous location: [nesterov_continuous_location.py](nesterov_continuous_location.py)
- piece-wise linear optimization: [nesterov_piecewise_linear.py](nesterov_piecewise_linear.py)
- sum of absolute values: [nesterov_sum_absolute.py](nesterov_sum_absolute.py)
- public entry file: [nesterov_smoothing.py](nesterov_smoothing.py)

The project currently implements:

- the matrix-game example from Sections 4.1 and 6
- the continuous location example from Section 4.2
- the piece-wise linear optimization example from Section 4.4.1
- the sum-of-absolute-values example from the paper's piece-wise linear discussion
- the shared accelerated scheme from Section 3

## Project layout

```text
.
|-- nesterov_continuous_location.py
|-- nesterov_core.py
|-- nesterov_matrix_game.py
|-- nesterov_piecewise_linear.py
|-- nesterov_sum_absolute.py
|-- nesterov_smoothing.py
|-- reproduce_continuous_location.py
|-- reproduce_nesterov_2005.py
|-- reproduce_piecewise_linear.py
|-- reproduce_sum_absolute.py
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

If you do not want to activate the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
```

## Running tests

```powershell
python -m unittest -v
```

## Running the demos

Run the built-in demos for all three examples:

```powershell
python nesterov_smoothing.py
```

## Matrix game example

Run one Section 6 instance:

```powershell
python reproduce_nesterov_2005.py --eps 0.01 --m-values 100 --n-values 100
```

Run the same instance with changing `mu`:

```powershell
python reproduce_nesterov_2005.py --eps 0.01 --m-values 100 --n-values 100 --continuation
```

Use the modified monotone `y_k` rule from the paper:

```powershell
python reproduce_nesterov_2005.py --eps 0.01 --m-values 100 --n-values 100 --monotone-y
```

Run the full Section 6 grid:

```powershell
python reproduce_nesterov_2005.py
```

Save the Section 6 results to CSV:

```powershell
python reproduce_nesterov_2005.py --csv results/section6.csv
```

## Continuous location example

Run the small built-in grid:

```powershell
python reproduce_continuous_location.py
```

Run one specific case:

```powershell
python reproduce_continuous_location.py --eps 0.1 --num-cities 10 --dimensions 2 --radius 1.0
```

Run it with changing `mu`:

```powershell
python reproduce_continuous_location.py --eps 0.1 --num-cities 10 --dimensions 2 --radius 1.0 --continuation
```

Save the continuous-location runs to CSV:

```powershell
python reproduce_continuous_location.py --csv results/continuous_location.csv
```

## Piece-wise linear example

Run the small built-in grid:

```powershell
python reproduce_piecewise_linear.py
```

Run one specific case:

```powershell
python reproduce_piecewise_linear.py --eps 0.1 --m-values 10 --n-values 5 --radius 1.0
```

Run it with changing `mu`:

```powershell
python reproduce_piecewise_linear.py --eps 0.1 --m-values 10 --n-values 5 --radius 1.0 --continuation
```

Save the piece-wise linear runs to CSV:

```powershell
python reproduce_piecewise_linear.py --csv results/piecewise_linear.csv
```

## Sum of absolute values example

Run the small built-in grid:

```powershell
python reproduce_sum_absolute.py
```

Run one specific case:

```powershell
python reproduce_sum_absolute.py --eps 0.1 --m-values 10 --n-values 5 --radius 1.0
```

Run it with changing `mu`:

```powershell
python reproduce_sum_absolute.py --eps 0.1 --m-values 10 --n-values 5 --radius 1.0 --continuation
```

Save the sum-of-absolute-values runs to CSV:

```powershell
python reproduce_sum_absolute.py --csv results/sum_absolute.csv
```

## Epsilon scaling experiment

To run the fixed-$\mu$ epsilon-scaling experiment for all four problems, use:

```powershell
python experiments/run_epsilon_scaling.py --output-dir results --overwrite
```

This experiment:

- uses the fixed paper-style smoothing parameter only,
- does not use continuation,
- reuses one random instance per seed across all epsilon values,
- runs seeds `0, 1, 2, 3, 4`,
- tests epsilon values `1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4`.

It writes:

- `results/epsilon_scaling.csv`
- `results/epsilon_scaling_summary.csv`
- `results/epsilon_scaling_regression.csv`
- `results/plots/epsilon_scaling_iterations_vs_inverse_epsilon.png`
- `results/plots/epsilon_scaling_loglog.png`

The regression file reports the estimated slope `p` in
`log(iterations) = beta0 + p * log(1 / epsilon)`.

## Notes

- The matrix-game and piece-wise linear examples both use entropy smoothing of a max-affine structure.
- The sum-of-absolute-values example uses quadratic smoothing of the `L1` objective's box-dual form.
- The continuous-location example uses quadratic smoothing of Euclidean distances.
- All three scripts support an optional continuation mode with `--continuation`, `--mu-start-factor`, `--mu-decay`, `--stage-factor`, and `--max-stages`.
- All three scripts also support the paper's monotone `y_k` acceptance rule through `--monotone-y`.
- `numpy` is required for the solvers, and `matplotlib` is required for the PNG plots created by the epsilon-scaling experiment.
- `unittest` is built into Python.

## Reference

- Paper PDF: https://webdoc.sub.gwdg.de/ebook/serien/e/CORE/dp2003-12.pdf

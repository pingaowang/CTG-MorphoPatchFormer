# Standalone WMFB comparator

This separate GPL-3.0 module provides `run_wmfb` and helpers for signal
interpolation and conversion of events to labels. It depends only on
NumPy and SciPy and does not import the first-party package.

Supply a prepared, finite FHR array in bpm. See [the module](wmfb.py) for
the interpolation helper's input requirements. `run_wmfb` returns event
times in seconds. To map them back to the original signal, use
`events_to_dense_labels` with that signal's timing information. This helper
includes the end point; the event evaluator uses half-open intervals.

Import the module directly from this directory or with Python's `importlib`.
It is not included in the first-party wheel. No preprocessing wrapper, training,
data download or clinical sample is supplied. The port is retained for method
inspection; MATLAB/Python numerical equivalence has not been established.

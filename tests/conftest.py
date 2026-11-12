import pathlib
import sys

# Import the circuit backend standalone as `circuits`: `discombo/__init__.py` eagerly imports all benchmarks,
# whose dependencies are not needed to test the circuits.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'discombo'))

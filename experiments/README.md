# Experiments

`steering_utils.py` provides space-token right padding, block-input capture, and
front injection for `colab-steering.ipynb`. The pilot uses red-minus-whitespace
with `c=5`, applied to the conditional row only. `steering_steps` selects the first
N evaluations, `"last"` only, or `"all"`.

"""Sentiment analysis project package.

`transformers` probes for TensorFlow when it is imported. In an environment that
has both TensorFlow and this project's pinned Keras 3 - which is exactly what
requirements/inference-cpu.txt installs, because the LSTM and GRU need Keras and
the transformer needs `transformers` - that probe raises:

    ValueError: Your currently installed version of Keras is Keras 3, but this
    is not yet supported in Transformers. Please install the backwards-
    compatible tf-keras package.

Nothing here uses the TensorFlow path inside `transformers`: the transformer
runs on PyTorch, and the Keras models are built directly against Keras. So the
probe is switched off here, before any submodule can import `transformers`.
Setting it this way avoids taking on the legacy tf-keras package as a
dependency purely to satisfy a code path the project never executes.
"""

import os

os.environ.setdefault("USE_TF", "0")

import logging

import numpy as np

logger = logging.getLogger(__name__)

ATTACK_NUMPY_DTYPE = np.float32  # pylint: disable=C0103
ATTACK_DATA_PATH: str
"""
ARC-1D (1D program search) data: child dataclass and loaders.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from boreft.task_config import task_config

from .base import ReftItem


@dataclass
class ArcItem(ReftItem):
    """ARC item: puzzle prompt, target output array."""

    task_type: str = ""  # task-type prefix (e.g. 'shift_right')
    filename: str = ""  # source puzzle filename

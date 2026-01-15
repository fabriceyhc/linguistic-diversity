#!/usr/bin/env python3
"""Create visualizations - with matplotlib backend fix."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Now import the rest
exec(open(Path(__file__).parent / "05_create_plots.py").read().replace(
    "import matplotlib.pyplot as plt\nimport seaborn as sns",
    "# Already imported"
))

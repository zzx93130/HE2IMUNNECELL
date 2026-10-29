import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm
from skimage.measure import regionprops

from cellpose import models

print("1,2,3,3,213,5413,51")
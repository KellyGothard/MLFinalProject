""" FileReader for us to draw in reddit posts and work with them"""

import numpy as np
import pandas as pd
import random

notbanned= pd.read_csv("notbanned.csv", delimiter=',')

pd.set_option('display.max_columns', 5)  # Set to actually print out the full columns, change if needed
print(notbanned.head(n=1))
for line in notbanned:
    print(line)

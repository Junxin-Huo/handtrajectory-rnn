import numpy as np
from loader import loadDataLabel, FRAME_COUNT, clipData
DATADIR = 'dataset_train'

data, label = loadDataLabel(DATADIR, shuffle=True, various=True)
data2, label2 = clipData(data, label)

print 'end'
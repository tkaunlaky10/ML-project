import json
import numpy as np
from matplotlib import pyplot as plt
inp = None
# with open('../Dataset_[timestamp]/CatA_Simple/Task000.json') as f:
# ''' 
#     Change the path to the dataset you want to use
# '''

with open('../Dataset/CatA_Simple/Task000.json') as f:
    inp = json.load(f)
    a = np.array(inp[0]['input'])
    b = np.array(inp[0]['output'])

plt.imshow(a)
plt.show()

plt.imshow(b)
plt.show()
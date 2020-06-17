import numpy as np
import matplotlib.pyplot as plt
from galaxylib import Galaxy

img  = np.random.rand(100,100)
meta = {"zp" : 0}

gal = Galaxy("1", size=30, ra=1, dec=1)

gal.add_data("test", "g", meta, img,   overwrite=True)
# gal.add_data("test", "g", meta, img+1, overwrite=True)
print("Done")

plt.imshow(gal.data["test/g/img"])
gal.close()

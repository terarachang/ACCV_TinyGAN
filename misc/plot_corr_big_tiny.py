import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
assert len(sys.argv) == 3, "tiny.npy, big.npy"

matplotlib.style.use('ggplot')
plt.xlim(0, 350)
plt.ylim(0, 350)

tiny = np.load(sys.argv[1])
big = np.load(sys.argv[2])
plt.scatter(tiny, big, color='black')

plt.plot(range(350), range(350), color='black')


plt.xlabel("TinyGAN", color='black', fontsize=18)
plt.ylabel("BigGAN", color='black', fontsize=18)

plt.savefig('tiny_big_correlation.png')

#plt.show()

from scipy import stats
r, _ = stats.pearsonr(tiny, big)
print('pearson r:',np.round(r, 2))

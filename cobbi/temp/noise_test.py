from noise import pnoise2, snoise2
import numpy as np
import matplotlib.pyplot as plt


octaves = 1
freq = 8. * octaves
max_y = 50
max_x = 40

for base in [0, 1, 2]:
    arr = np.zeros((max_y, max_x))
    for y in range(max_y):
        for x in range(max_x):
            arr[y, x] = pnoise2(x / freq, y / freq, octaves, base=base)
    plt.figure()
    plt.imshow(arr)
    plt.show()
    print('RMSE: {:g}'.format(np.sqrt(np.mean(arr**2))))

print('end')
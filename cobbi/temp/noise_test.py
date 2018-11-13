from noise import pnoise2, snoise2
import numpy as np
import matplotlib.pyplot as plt


octaves = 4
freq = 8. * octaves
max_y = 50
max_x = 40

plt.ioff()
for base in [0, 1, 2, 3, 4]:
    for freq in [12, 8, 6, 4]:
        arr = np.zeros((max_y, max_x))
        for y in range(max_y):
            for x in range(max_x):
                arr[y, x] = pnoise2(x / freq, y / freq, octaves, base=base)
        plt.figure()
        plt.imshow(arr)
        plt.title('base {:g} freq {:g}'.format(base, freq))
        plt.show()
        print('RMSE: {:g}'.format(np.sqrt(np.mean(arr**2))))

plt.ion()

print('end')
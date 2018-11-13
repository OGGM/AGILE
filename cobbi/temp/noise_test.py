from noise import pnoise2, snoise2
import numpy as np
import matplotlib.pyplot as plt


octaves = 4
freq = 8. * octaves
max_y = 50
max_x = 40
# extent = (0, max_x, 0, max_y)

plt.ioff()
for base in [0, 1, 2, 3, 4]:
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)

    for i, freq in enumerate([12, 8, 6, 4]):
        arr = np.zeros((max_y, max_x))
        for y in range(max_y):
            for x in range(max_x):
                arr[y, x] = pnoise2(x / freq, y / freq, octaves, base=base)
        # scale:
        arr *= min(1./arr.min(), 1./arr.max())
        im = axs[i].imshow(arr)
        #axs[i].set_title('base {:g} freq {:g}'.format(base, freq))
        axs[i].label_outer()
        print('RMSE: {:g}'.format(np.sqrt(np.mean(arr**2))))

    fig.colorbar(im, ax=axs, orientation='horizontal')
    #plt.tight_layout()
    plt.show()

plt.ion()

print('end')
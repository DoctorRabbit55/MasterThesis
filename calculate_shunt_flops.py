from CDL.shunt import Architectures
from CDL.utils.calculateFLOPS import calculateFLOPs_model

from matplotlib import pyplot as plt

if __name__ == '__main__':

    arch = 1

    MAccs_list = []

    for width_height in range(4, 15):

        shunt = Architectures.createShunt((width_height, width_height, 64), (width_height, width_height, 96), arch)
        MAccs_list.append(calculateFLOPs_model(shunt)['total'])

    print(MAccs_list)
    plt.plot(range(4,15), MAccs_list)
    plt.show()
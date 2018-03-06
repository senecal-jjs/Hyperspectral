from image import HyperCube
import pickle
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy 


if __name__ == '__main__':
    imager_wavelengths = np.array([
            387.12, 389.13, 391.13, 393.13, 395.14, 397.14, 399.15, 401.16, 403.17, 405.17,
            407.18, 409.19, 411.21, 413.22, 415.23, 417.25, 419.26, 421.28, 423.29, 425.31,
            427.33, 429.35, 431.37, 433.39, 435.41, 437.43, 439.46, 441.48, 443.51, 445.53,
            447.56, 449.59, 451.62, 453.64, 455.68, 457.71, 459.74, 461.77, 463.8, 465.84,
            467.87, 469.91, 471.95, 473.98, 476.02, 478.06, 480.1, 482.14, 484.19, 486.23,
            488.27, 490.32, 492.36, 494.41, 496.46, 498.5, 500.55, 502.6, 504.65, 506.7,
            508.76, 510.81, 512.86, 514.92, 516.97, 519.03, 521.09, 523.15, 525.21, 527.26,
            529.33, 531.39, 533.45, 535.51, 537.58, 539.64, 541.71, 543.77, 545.84, 547.91,
            549.98, 552.05, 554.12, 556.19, 558.26, 560.34, 562.41, 564.49, 566.56, 568.64,
            570.72, 572.79, 574.87, 576.95, 579.04, 581.12, 583.2, 585.28, 587.37, 589.45,
            591.54, 593.63, 595.71, 597.8, 599.89, 601.98, 604.07, 606.16, 608.26, 610.35,
            612.45, 614.54, 616.64, 618.73, 620.83, 622.93, 625.03, 627.13, 629.23, 631.33,
            633.44, 635.54, 637.65, 639.75, 641.86, 643.96, 646.07, 648.18, 650.29, 652.4,
            654.51, 656.63, 658.74, 660.85, 662.97, 665.08, 667.2, 669.32, 671.44, 673.55,
            675.67, 677.8, 679.92, 682.04, 684.16, 686.29, 688.41, 690.54, 692.66, 694.79,
            696.92, 699.05, 701.18, 703.31, 705.44, 707.57, 709.71, 711.84, 713.98, 716.11,
            718.25, 720.39, 722.53, 724.67, 726.81, 728.95, 731.09, 733.23, 735.38, 737.52,
            739.66, 741.81, 743.96, 746.11, 748.25, 750.4, 752.55, 754.71, 756.86, 759.01,
            761.16, 763.32, 765.47, 767.63, 769.79, 771.95, 774.1, 776.26, 778.42, 780.59,
            782.75, 784.91, 787.08, 789.24, 791.41, 793.57, 795.74, 797.91, 800.08, 802.25,
            804.42, 806.59, 808.76, 810.93, 813.11, 815.28, 817.46, 819.64, 821.81, 823.99,
            826.17, 828.35, 830.53, 832.71, 834.89, 837.08, 839.26, 841.45, 843.63, 845.82,
            848.01, 850.2, 852.39, 854.58, 856.77, 858.96, 861.15, 863.34, 865.54, 867.73,
            869.93, 872.13, 874.32, 876.52, 878.72, 880.92, 883.12, 885.33, 887.53, 889.73,
            891.94, 894.14, 896.35, 898.56, 900.76, 902.97, 905.18, 907.39, 909.6, 911.82,
            914.03, 916.24, 918.46, 920.67, 922.89, 925.11, 927.32, 929.54, 931.76, 933.98,
            936.21, 938.43, 940.65, 942.87, 945.1, 947.33, 949.55, 951.78, 954.01, 956.24,
            958.47, 960.7, 962.93, 965.16, 967.39, 969.63, 971.86, 974.1, 976.34, 978.57,
            980.81, 983.05, 985.29, 987.53, 989.77, 992.02, 994.26, 996.5, 998.75, 1001.0,
            1003.24, 1005.49, 1007.74, 1009.99, 1012.24, 1014.49, 1016.74, 1018.99, 1021.25, 1023.5
        ])

    # Load spectra
    try:
        produce_spectra = pickle.load( open( "potato1.p", "rb" ))
    
    except (OSError, IOError) as e:
        produce_spectra = []
    
    reflectances = [item[:290] for item in produce_spectra]
    labels = [int(item[290]) for item in produce_spectra]

    plt.plot(reflectances[1], label=1)
    plt.plot(reflectances[4], label=2)
    plt.plot(reflectances[7], label=3)
    plt.plot(reflectances[10], label=6)
    plt.plot(reflectances[13], label=7)
    plt.plot(reflectances[16], label=10)
    plt.plot(reflectances[19], label=14)
    #plt.plot(reflectances[21], label=5)
    plt.legend(loc='upper left')
    #plt.show()
    
    # Assign colors to labels
    #color_options = ['cyan', 'pink', 'magenta', 'red', 'blue', 'green', 'black']
    #b_colors = {1: 'red', 2: 'red', 3: 'red', 5: 'green', 6: 'green', 7: 'green', 9: 'black', 10: 'black', 13: 'black', 15: 'black'}
    colors = {1: 'red', 2: 'green', 3: 'green', 6: 'cyan', 7: 'cyan', 10: 'black', 14: 'black'}
    pt_colors = []
    for label in labels:
        # print(label)
        # if label not in colors:
        #     colors[label] = color_options.pop()
        pt_colors.append(colors[label])

    pt_colors = ['red','red','blue','blue','black','black','black']
    # Use the first spectra as baseline
    baseline = reflectances[1]

    div = []
    corr = []
    area = []
    for i in range(1, len(reflectances), 3):
        div.append(utils.spectral_info_divergence(baseline, reflectances[i]))
        corr.append(utils.spectral_correlation(baseline, reflectances[i]))
        area.append(np.trapz(reflectances[i], imager_wavelengths[1:291]))

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(div, corr, area, color=pt_colors)
    plt.show()
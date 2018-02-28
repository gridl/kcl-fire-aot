import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator

def planck_radiance(wvl, temp):
    '''
    wvl: wavelngth (microns)
    temp: temperature (kelvin)
    '''
    c1 = 1.19e-16  # W m-2 sr-1
    c2 = 1.44e-2  # mK
    wt = (wvl*1.e-6) * temp # m K
    d = (wvl*1.e-6)**5 * (np.exp(c2/wt)-1)
    return c1 / d * 1.e-6  # W m-2 sr-1 um-1


# set up some T's and some flares areas and also pixel area
wavelengths = [4.0, 8.4]  # microns
lambda_max = [2900 / 4, 2900 / 8.4]
temps = np.arange(300, 1300, 1)   # in K
opt_temp = [722, 400]
labels = [ "4 $\mu m$", "8.4 $\mu m$"]
dashes = [(2.5, 2.5), (4.5, 4.5)]
fire_size = 1000   # in sq. m.
pixel_size = 4e6  # in sq. m.
flare_area_pc = fire_size / pixel_size  # unitless
frp_true = pixel_size * flare_area_pc * (const.sigma * temps ** 4)

# ====== Plot ====== #

# Set up figure and image grid
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

for wvl, opt_t, label, dash in zip(wavelengths, opt_temp, labels, dashes):

    l_true = flare_area_pc * planck_radiance(wvl, temps)

    a = planck_radiance(wvl, opt_t) / opt_t**4
    frp_assumed = ((pixel_size * const.sigma / a) * l_true) # in MW

    bias = (frp_assumed - frp_true) / frp_true * 100
    ax.plot(temps, bias, "k", label=label, linewidth=2, dashes=dash)

    ax.legend()
    ax.set_xlabel("Temperature (K)", fontsize=16)
    ax.set_ylabel("% FRP Difference", fontsize=16)

    ax.set_ylim(-50, 50)
    ax.set_xlim(300, 1300)

    ones = np.ones(temps.size)
    ax.plot(temps, np.ones(temps.size)*0, "k", linewidth=1)


    major_ticks = np.arange(-50, 51, 10)
    minor_ticks = np.arange(-50, 51, 5)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    #ax.tick_params(which='both', direction='out', labelright=True)
    ax.grid(which='minor',color='k', alpha=0.2, axis='y')
    ax.grid(which='major',color='k', alpha=0.5, axis='y')

    mean = np.mean(bias[(temps >= 350) & (temps < 500)])
    std = np.std(bias[(temps >= 350) & (temps < 500)])
    max = np.max(np.abs(bias[(temps >= 350) & (temps < 500)]))
    print "mean bias for band over smouldering Temps", label, ":", mean
    print "std bias for band over smouldering Temps", label, ":", std
    print "abs max band over smoulering Temps", label, ":", max
    print

    mean = np.mean(bias[(temps >= 700) & (temps < 1000)])
    std = np.std(bias[(temps >= 700) & (temps < 1000)])
    max = np.max(np.abs(bias[(temps >= 700) & (temps < 1000)]))
    print "mean bias for band over flaming Temps", label, ":", mean
    print "std bias for band over flaming Temps", label, ":", std
    print "abs max band over flaming Temps", label, ":", max
    print
    print

    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((350,-50), 150, 100,
                                    facecolor='red', edgecolor='none',
                                    alpha=0.15))

    currentAxis.add_patch(Rectangle((700, -50), 300, 100,
                                    facecolor='gold', edgecolor='none',
                                    alpha=0.15))


    # now compute the percent change over the flare range from 1600K to 2000K
    # min_ratio = np.min(ratio[(temp >= 1600) & (temp < 2200)])
    # max_ratio = np.max(ratio[(temp >= 1600) & (temp < 2200)])
    # percent_change = (max_ratio - min_ratio) / min_ratio * 100
    # print "percent change in ratio for band", label, ":", percent_change
    #
    # # print the max ratio error at the optimum temperature
    # err = frp_assumed[temp_mask, opt_temp] / frp_true[temp_mask, opt_temp]
    # print wvl, 'max error pecentage:', np.max(np.abs(err-1))


# plt.savefig('/Users/danielfisher/Dropbox/working_documents/papers/TGRS-swir-frp/figures/round_2/Figure_3.jpg',
#             bbox_inches='tight', dpi=600)
plt.show()



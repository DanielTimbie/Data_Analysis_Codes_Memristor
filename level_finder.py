import numpy as np
import os


# finds the levels to be used for programming
def rangefind(rmin, rmax, level_num):
    S_min = rmax ** (-1)
    S_max = rmin ** (-1)
    level_step = (S_max - S_min) / level_num
    level_gap = level_step / 2
    s_counter = 0
    r_array = []
    while s_counter < level_num:
        # s_array = s_array + [(S_min + level_step * s_counter, S_min + level_step * s_counter + level_gap)]
        r_array = r_array + [
            ((S_min + level_step * s_counter) ** (-1), (S_min + level_step * s_counter + level_gap) ** (-1))]
        s_counter += 1
    print(r_array)
    return r_array


rangefind(8e6, 1e10, 10)

# dynamic resistance range 1e7 - 1e9, want to get this in terms of conductance
# dynamic conductance range 1nS - 1000nS
# 1nS, 100nS, 200nS, 300nS, 400nS, 500nS, 600nS, 700nS, 800nS, 900nS
# 1GΩ, 10MΩ, 5mΩ, 3.33MΩ, 2.5MΩ, 2MΩ, 1.66MΩ, 1.43MΩ, 1.25MΩ, 1.11MΩ
# temp = 1e7-1.2e7, 1.5e7-1.97e7, 2.87e7-4.64e7, 1.8e8-5e9
# 1.6e-8 in between levels
# 1.3e-8 in between vals

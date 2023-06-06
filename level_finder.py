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


rangefind(1e7, 1e10, 50)

# dynamic resistance range 1e7 - 1e9, want to get this in terms of conductance
# dynamic conductance range 1nS - 1000nS
# 1nS, 100nS, 200nS, 300nS, 400nS, 500nS, 600nS, 700nS, 800nS, 900nS
# 1GΩ, 10MΩ, 5mΩ, 3.33MΩ, 2.5MΩ, 2MΩ, 1.66MΩ, 1.43MΩ, 1.25MΩ, 1.11MΩ
# temp = 1e7-1.2e7, 1.5e7-1.97e7, 2.87e7-4.64e7, 1.8e8-5e9
# 1.6e-8 in between levels
# 1.3e-8 in between vals

#9.1e8, 1e10
#2.44e7, 2.5e7

# (10000000000.0, 909918107.3703367)
#             (244140625.0, 196270853.77821392)
# (123578843.3020267, 109998900.0109999)
#             (82726671.0787558, 76411706.2734011)
# (62173588.65953743, 58537727.565415904)
#             (49800796.812749006, 47440580.67270743)
# (41535138.72736335, 39880358.923230305)
#             (35622684.52550584, 34398541.501840316)
# (31183734.56405139, 30241630.6287235)
#             (27728482.6974268, 26981086.25853277)
# (24962556.165751375, 24355196.18110524)

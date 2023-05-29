import model
import math
import numpy as np
import pandas as pd


import evaluation as ev

if __name__ == "__main__":
    TOTALPIXEL = 15 # CHANGE
    SLITNUM = 4 # The number of slits each pixel has
    example_image = [[1, 1, 1],[1, 0, 1],[1, 0, 1],[1, 0, 1],[1, 1, 1]]
    
    df = model.random_model_generator(TOTALPIXEL, SLITNUM, example_image)
    #df = pd.read_csv("sample_models/model1.csv")

    L_01:float = 100*math.pow(10,-2)
    L_12:float = 100*math.pow(10,-2)
    L_23:float = 100*math.pow(10,-2)
    centerpos1 = df.loc[df['LayerIndex'] == 1,'CenterPosition'].values
    centerpos2 = df.loc[df['LayerIndex'] == 2,'CenterPosition'].values
    delta_a_j:float = math.pow(10,-5)
    beta1 = df.loc[df['LayerIndex'] == 1,'Beta'].values
    beta2 = df.loc[df['LayerIndex'] == 2,'Beta'].values

    #--------------------------------------
    # TO DO: Bunu daha generik bir sekilde yazmam gerekiyor
    y_target = [0.00000001]*1001
    for i in range(501):
        if i < 20:
            y_target[i] = 1
    y_target = np.array(y_target)
    #--------------------------------------
    psi_N_conj = np.conj(ev.psi_N(ev.g_zero, ev.g_one_or_g_two, 
                                  ev.g_one_or_g_two, centerpos1, centerpos2, 
                                  delta_a_j,  L_01, L_12, L_23, beta1,beta2)[3])
    a,y_pred,b,c = ev.psi_N(ev.g_zero, ev.g_one_or_g_two, ev.g_one_or_g_two,
                            centerpos1, centerpos2, delta_a_j, L_01, L_12, L_23,
                            beta1, beta2)
    #--------------------------------------

    model.optimize(psi_N_conj, centerpos1, centerpos2, delta_a_j, L_01, L_12, L_23, beta1, beta2, y_pred, y_target)

import math
import random
import numpy as np
import pandas as pd
import evaluation as ev
from sklearn.linear_model import LogisticRegression

def random_model_generator(total_pixel:int, slit_num:int, image:list):
    # Max beta value is 50 MICRO and beta is set to max value.
    # center position of slits must be between beta and 2 beta
    """
    total_pixel: Total pixel number of image
    slit_num: Assigned slit number for each pixel of image
    """ 
    MICRO = math.pow(10, -6)
    total_slit_number = list(range(1, (total_pixel * slit_num) + 1))
    beta = [50] * len(total_slit_number)
    centerpos = list(range(25, (2 * len(total_slit_number) * 50) + 25, 100))
    pixel_no = total_slit_number[:len(total_slit_number) // 4] * slit_num
    pixel_value = [0, 0, 1, 1]

    # Generating layer parameters 
    def generate_layer_data(layer_no):
        random.shuffle(pixel_no)
        layer_data = {'LayerIndex': [layer_no] * len(total_slit_number),
                      'SlitNo': total_slit_number,
                      'CenterPosition': centerpos,
                      'Beta': beta,
                      'PixelNo': pixel_no
                      }
        df = pd.DataFrame(layer_data).sort_values(by=['PixelNo'])
        df['PixelValue'] = pixel_value * int(len(total_slit_number) / 4)
        df = df.sort_values(by=['SlitNo'])
        df['CenterPosition'] *= MICRO
        df['Beta'] *= MICRO
        return df
    
    first_df = generate_layer_data(1)
    second_df = generate_layer_data(2)

    image = np.array(image).flatten() 
    image_data = {'PixelNo': list(range(1, len(image) + 1)),
                  'PixelValue': image.tolist()
                  }
    image_df = pd.DataFrame(image_data)

    first_df = pd.merge(first_df, image_df, on=['PixelNo', 'PixelValue'], 
                        how='inner').sort_values(by=['SlitNo'], 
                                                 ignore_index=True)
    second_df = pd.merge(second_df, image_df, on=['PixelNo', 'PixelValue'], 
                         how='inner').sort_values(by=['SlitNo'], 
                                                  ignore_index=True)

    result_df = pd.concat([first_df, second_df])
    return result_df

def calculate_loss_values(centerpos1: list, centerpos2: list, delta_a_j: float, 
            L_01: float, L_12: float, L_23: float, beta1: list, beta2: list, 
            y_target) -> float:
    # Equation 8 in report
    # forward propagation calculation
    a_i, y_pred, energy, psi = ev.psi_N(ev.g_zero, ev.g_one_or_g_two,
                                        ev.g_one_or_g_two, centerpos1, 
                                        centerpos2, delta_a_j, L_01, L_12, L_23,
                                        beta1, beta2)
    
    # Calculate difference between target and predict
    loss = ev.calculate_error(y_pred, y_target)
    return loss 

def calculate_derror_over_dL(psi_N_conj:complex, func_g0, func_g1, func_g2, 
                             centerpos1:list, centerpos2:list, delta_a_j:float, 
                             L_01:float, L_12:float, L_23:float, beta1:list, 
                             beta2:list, y_pred:list, y_target:list) -> float:
    # Equation 10 in report
    dy_pred_over_dL = ev.dy_pred_over_dL(psi_N_conj, func_g0, func_g1, func_g2, 
                                         centerpos1, centerpos2, delta_a_j, 
                                         L_01, L_12, L_23, beta1, beta2)
    result = 2* sum((y_pred - y_target) * dy_pred_over_dL) /len(y_pred)

    return result

def calculate_derror_over_dbeta(psi_N_conj:complex, fun_dpsi_over_dbeta, 
                                centerpos1, centerpos2, delta_a_j:float, 
                                L_01:float, L_12:float, L_23:float, beta1, 
                                beta2, y_pred:list, y_target:list ) -> float:
    # Equation 9 in report
    dy_pred_over_dbeta = ev.dy_pred_over_dbeta(psi_N_conj, fun_dpsi_over_dbeta, 
                                               centerpos1, centerpos2, 
                                               delta_a_j, L_01, L_12, L_23, 
                                               beta1, beta2)
    result = 2* sum((y_pred - y_target) * dy_pred_over_dbeta) /len(y_pred)

    return result

def optimize(digit, psi_N_conj, centerpos1: list, centerpos2: list, 
             delta_a_j: float, L_01: float, L_12: float, L_23: float,
             beta1: list, beta2: list, y_pred, y_target):
    epoch = 0
    error = 999

    errors = list()
    epochs = list()

    learning_rate = 0.01
    opt_df = pd.DataFrame(columns=['Image digit', 'Learning rate', 'Epoch', 
                                   'loss', 'L_01', 'L_12', 'L_23', 'Beta1', 
                                   'Beta2'])
    while (epoch <= 50) and (error > 9e-4):

        loss = calculate_loss_values(centerpos1, centerpos2, delta_a_j, L_01, 
                                     L_12, L_23, beta1, beta2, y_target)
         
        loss_dL01 = calculate_derror_over_dL(psi_N_conj, ev.dfunction_for_L01, 
                                             ev.g_one_or_g_two, 
                                             ev.g_one_or_g_two, centerpos1, 
                                             centerpos2, delta_a_j, L_01, L_12, 
                                             L_23, beta1, beta2, y_pred, 
                                             y_target)        
        L_01 = L_01 - (learning_rate * loss_dL01)

        loss_dL12 = calculate_derror_over_dL(psi_N_conj, ev.g_zero, 
                                        ev.dfunction_for_L12_or_L23, 
                                        ev.g_one_or_g_two, centerpos1, 
                                        centerpos2, delta_a_j, L_01, L_12, 
                                        L_23, beta1, beta2, y_pred, 
                                        y_target)  
        L_12 = L_12 - (learning_rate * loss_dL12)

        loss_dL23 = calculate_derror_over_dL(psi_N_conj, ev.g_zero, 
                                    ev.g_one_or_g_two, 
                                    ev.dfunction_for_L12_or_L23, centerpos1, 
                                    centerpos2, delta_a_j, L_01, L_12, 
                                    L_23, beta1, beta2, y_pred, 
                                    y_target)  
        L_23 = L_23 - (learning_rate * np.array(loss_dL23))
        
        for i, beta in enumerate(beta1):
            loss_db1 = calculate_derror_over_dbeta(psi_N_conj, 
                                                   ev.dpsi_over_dbeta1, 
                                                   centerpos1[i], centerpos2, 
                                                   delta_a_j, L_01, L_12, L_23, 
                                                   beta, beta2, y_pred,y_target)  
            beta1[i] = beta - (learning_rate * loss_db1)
        
        for i, beta in enumerate(beta2):
            loss_db2 = calculate_derror_over_dbeta(psi_N_conj, 
                                                   ev.dpsi_over_dbeta2, 
                                                   centerpos1, centerpos2[i], 
                                                   delta_a_j, L_01, L_12, L_23, 
                                                   beta1, beta, y_pred,y_target)  
            beta2[i] = beta - (learning_rate * loss_db2)

        loss = (learning_rate * loss)

        errors.append(loss)
        epochs.append(epoch)
        error = errors[-1]
        epoch += 1

        
        opt_result = {'Image digit': digit, 'Learning rate': learning_rate, 
                      'Epoch': epoch, 'loss': errors[-1], 'L_01': L_01, 
                      'L_12': L_12, 'L_23': L_23, 'Beta1': beta1, 
                      'Beta2': beta2}

        print('Learning rate: {} Epoch {}. loss: {}'.format(learning_rate, 
                                                            epoch, errors[-1]))
    
    opt_df.append(opt_result)
    return opt_result

def train_test(total_pixel, slit_num, image:dict, y_target:dict):

    # For first step
    df = random_model_generator(total_pixel, slit_num, image.get(0))
    df.to_csv('model_for_{}.csv'.format(0))
    L_01:float = 100*math.pow(10,-2)
    L_12:float = 100*math.pow(10,-2)
    L_23:float = 100*math.pow(10,-2)
    centerpos1 = df.loc[df['LayerIndex'] == 1,'CenterPosition'].values
    centerpos2 = df.loc[df['LayerIndex'] == 2,'CenterPosition'].values
    delta_a_j:float = math.pow(10,-5)
    beta1 = df.loc[df['LayerIndex'] == 1,'Beta'].values
    beta2 = df.loc[df['LayerIndex'] == 2,'Beta'].values

    a_i, y_pred, energy, psi_N = ev.psi_N(ev.g_zero, ev.g_one_or_g_two, 
                                          ev.g_one_or_g_two,
                                          centerpos1, centerpos2, delta_a_j, 
                                          L_01, L_12, L_23, beta1, beta2)
    psi_N_conj = np.conj(psi_N)
    opt_result = optimize(0, psi_N_conj, centerpos1, centerpos2, delta_a_j, 
                          L_01, L_12, L_23, beta1, beta2, y_pred, 
                          y_target.get(0))

    #For other images
    for i in range(1, 10):
        df = random_model_generator(total_pixel, slit_num, image.get(i))
        df.to_csv('model_for_{}.csv'.format(i))
        L_01:float = opt_result['L_01'].iloc[-1]
        L_12:float = opt_result['L_12'].iloc[-1]
        L_23:float = opt_result['L_23'].iloc[-1]
        centerpos1 = df.loc[df['LayerIndex'] == 1,'CenterPosition'].values
        centerpos2 = df.loc[df['LayerIndex'] == 2,'CenterPosition'].values
        delta_a_j:float = math.pow(10,-5)
        beta1 = opt_result['Beta1'].iloc[-1]
        beta2 = opt_result['Beta2'].iloc[-1]

        a_i, y_pred, energy, psi_N = ev.psi_N(ev.g_zero, ev.g_one_or_g_two, 
                                              ev.g_one_or_g_two, centerpos1, 
                                              centerpos2, delta_a_j, L_01, L_12,
                                              L_23, beta1, beta2)
        psi_N_conj = np.conj(psi_N)
        opt_result = optimize(i, psi_N_conj, centerpos1, centerpos2, delta_a_j,
                              L_01, L_12, L_23, beta1, beta2, y_pred, 
                              y_target.get(i))
    
    return sum(opt_result['loss']) / len(opt_result['loss'])
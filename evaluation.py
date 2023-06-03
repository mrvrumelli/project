import math, cmath
import numpy as np

S = 2 * 10**(-3) # sigma value
Lambda_ = 635 * 10**(-9) # Lambda_bda value
PI = math.pi # pi value

def round_number(number: float) -> float:
    # Function for controlling floating numeric point. 
    # In some case floating points error can occur(ex 2.00000000013 for 2) 
    k = 1 - int(math.log10(abs(number)))
    if number < 0:
        floating_number = round(number, None if -number > 1 else k)
    else:
        floating_number = round(number, None if number > 1 else k)
        
    return floating_number 

def activation_function(psi):
    """
    psi: prediction output
    """ 
    return abs(psi)**2

def g_zero(centerpos1: float, L_01: float) -> complex:
    # Equation 4 in report 
    # Returns value between input and first layer (g0)
    """
    centerpos1: Slit position of first layer as input
    L_01: Distance between input and first layer
    """
    # Numerator and denominator for the root term calculation
    numerator = 2 * math.sqrt(PI) * S
    denominator = (Lambda_ * L_01) - (2 * 1j * PI * (S**2))

    # Exponential term calculation
    exponential_term = cmath.exp(
        (-2 * (PI**2) * (S**2) * (centerpos1**2)) / (4 * (PI**2) * S**4 
        + (Lambda_ * L_01)**2)+ 1j * centerpos1**2 * (PI / Lambda_ * L_01 
        - (4 * (PI**3) * S**4 / Lambda_ * L_01 * (4 * (PI**2) * (S**4) 
        + (Lambda_ * L_01)**2)))- 1j * PI / 4)
    
    # Calculate the root term
    root_term = (numerator / denominator)**(1/2)

    # Return value of calculation
    return root_term * exponential_term

def g_one_or_g_two(centerpos1:float, centerpos2:float, L_value:float, 
                   beta:float) -> complex:
    # Equations 5, 6 in report 
    # Returns value between first layer and second layer (g1) or,
    # Returns value between second layer and last layer (g2)
    """
    centerpos1: Current slit position as input
    centerpos2: Slit position on next layer 
    L_value: Distance between two layers
    beta: Slit width
    """
    numerator = (1 - 1j) * cmath.exp((-1 * PI * (centerpos1 - centerpos2)**2) 
                            / ((2 * PI * beta**2) + (1j * Lambda_ * L_value)))
    denominator = (-2j) + ((Lambda_ * L_value) / (PI * beta**2))
    
    return (numerator / denominator)**(1/2)

def calculate_error(y_pred, y_target):
    """
    y_pred: Predicted value after processing
    y_target: Desired value after processing
    """
    return np.sum(np.square(y_target - y_pred))/ len(y_target)

def dfunction_for_L01(centerpos:float, L:float) -> complex:
    # Equation 30 in report 
    """
    centerpos: Center position of slits on first layer
    L: Distance between input and first layer(L01)
    """
    term1 = (-1) ** (3/4) * (PI ** (1/4)) * Lambda_ * S
    term2 = 1 + 2j * centerpos**2 * (2 * PI * S**2 + 1j * Lambda_ * L) * (
            (4 * PI**2 * Lambda_ * L * S**2) / (Lambda_**2 * L**2 + 4 *\
            PI * S**4)**2 + (1j * (4 * PI**3 * S**4 - PI * Lambda_**2 * L**2)) / 
            (Lambda_**2 * L**2 + 4 * PI**2 * S**4)**2
    )
    term3 = cmath.exp(PI * centerpos**2 * (
            (-2 * PI * S**2) / (Lambda_**2 * L**2 + 4 * PI * S**4) +
            (1j * Lambda_ * L) / (Lambda_**2 * L**2 + 4 * PI**2 * S**4)
    ))
    term4 = math.sqrt(2) * (2 * PI * S**2 + 1j * Lambda_ * L)**2
    term5 = cmath.sqrt(S / (Lambda_ * L - 2j * PI * S**2))

    result = term1 * term2 * term3 / (term4 * term5)
    return result

def dfunction_for_L12_or_L23(centerpos1:float, centerpos2:float, L:float, 
                             beta:float) -> complex:
    # Equation 35 in report 
    """
    centerpos1: Center position of slits on current layer
    centerpos2: Center position of slits on next layer
    L: Distance between current layer and previous layer
    beta: With of each slit on current layer
    """    
    term1 = ((1/2) + (1j/2)) * Lambda_ * cmath.exp((-PI * (centerpos1 -\
            centerpos2)**2) / (2 * PI * beta**2 + 1j *Lambda_ * L))
    term2 = (2 * PI * (beta + centerpos1 - centerpos2) * 
             (beta - centerpos1 + centerpos2) + 1j * Lambda_ * L)
    term3 = (2 * PI * beta**2 + 1j * Lambda_ * L)**2
    term4 = cmath.sqrt((Lambda_ * L / (PI * beta**2)) - 2j)

    result = term1 * term2 / (term3 * term4)
    return result

def dfunction_for_B1_or_B2(centerpos1:float, centerpos2:float, L:float, 
                             beta:float) -> complex:
    # Equation 39 in report 
    """
    centerpos1: Center position of slits on current layer
    centerpos2: Center position of slits on next layer
    L: Distance between current layer and previous layer
    beta: With of each slit on current layer
    """    
    term1 = (1 - 1j) * cmath.exp((-PI * (centerpos1 - centerpos2)**2) / 
                                (2 * PI * beta**2 + 1j * Lambda_ * L))
    term2 = (Lambda_**2 * L**2 - 2 * PI * beta**2 * (2 * PI * 
            (centerpos1 - centerpos2)**2 + 1j * Lambda_ * L))
    term3 = beta * (2 * PI * beta**2 + 1j * Lambda_ * L)**2
    term4 = cmath.sqrt((Lambda_ * L / (PI * beta**2)) - 2j)

    result = term1 * term2 / (term3 * term4)
    return result

def psi_N(func_g0, func_g1, funcg2, centerpos1: list, centerpos2: list, 
          delta_a_j: float, L_01: float, L_12: float, L_23: float, beta1: list, 
          beta2: list): 
    # y_pred = abs(psi)**2
    # Returns predicted values for each point in measurement layer
    # Equations 7, 18, 20, 22 in report 
    """
    centerpos1: Slit position for first layer
    centerpos2: Slit position for second layer
    delta_a_j: Size of a_i
    L_01: Distance between input and first layer
    L_12: Distance between first and second layer
    L_23: Distance between second and third layer
    beta1: Widths in first layer slits
    beta2: Widths in second layer slits
    """
    a_j = [i * delta_a_j  for i in range(-500, 501)]
    y_pred = []
    psi = []
    
    for a in a_j:
        result = 0
        for j, pos2 in enumerate(centerpos2):
            centerpos2[j] = round_number(pos2)
            sum_g0_g1 = sum(func_g0(round_number(pos1), L_01) *\
                            func_g1(round_number(pos1),centerpos2[j], L_12, 
                                    beta) for pos1, beta in zip(centerpos1, 
                                                                beta1))
            g2 = funcg2(centerpos2[j], a, L_23, beta2[j])
            result += sum_g0_g1 * g2

        psi.append(result) 
        y_pred.append(activation_function(result))
    
    a_j = [item * 10**3 for item in a_j]
    total_sum = sum(y_pred) * delta_a_j
    return a_j, y_pred, total_sum, psi

def dy_pred_over_dL(psi_N_conj, func_g0, func_g1, funcg2, 
                        centerpos1: list, centerpos2: list, delta_a_j: float, 
                        L_01: float, L_12: float, L_23: float, beta1: list, 
                        beta2: list):
    # Equation 16 in report
    """
    psi_N_conj: Conjugate of psi equation 
    func_g0: function of g0, if y_pred over dL01 is calculated this fuction is
             dfunction_for_L01
    func_g1: function of g1, if y_pred over dL12 is calculated this fuction is
             dfunction_for_L12_or_L23
    funcg2: function of g2, if y_pred over dL23 is calculated this fuction is
             dfunction_for_L12_or_L23
    centerpos1: Slit position for first layer
    centerpos2: Slit position for second layer
    delta_a_j: Size of a_i
    L_01: Distance between input and first layer
    L_12: Distance between first and second layer
    L_23: Distance between second and third layer
    beta1: Widths in first layer slits
    beta2: Widths in second layer slits
    """   
    dpsi_N = psi_N(func_g0, func_g1, funcg2,centerpos1, centerpos2, delta_a_j,
                   L_01, L_12, L_23, beta1,beta2)[3]
    result = 2* (dpsi_N * psi_N_conj)
    return result.real

def dpsi_over_dbeta1(centerpos1: float, centerpos2: list, delta_a_j: float, 
                        L_01: float, L_12: float, L_23: float, beta1: float, 
                        beta2: list):
    # Equation 23 in report
    """
    centerpos1: Slit position effected by calculated beta 
    centerpos2: Slit position for second layer
    delta_a_j: Size of a_i
    L_01: Distance between input and first layer
    L_12: Distance between first and second layer
    L_23: Distance between second and third layer
    beta1: Calculated width in first layer
    beta2:  Widths in second layer slits
    """   
    a_j = [i * delta_a_j  for i in range(-500, 501)]
    dpsi = []
    
    for a in a_j:
        result = 0
        for j, pos2 in enumerate(centerpos2):
            centerpos2[j] = round_number(pos2)
            g0 = g_zero(centerpos1, L_01)
            dg1 = dfunction_for_B1_or_B2(centerpos1, centerpos2[j], L_12, beta1)
            g2 = g_one_or_g_two(centerpos2[j], a, L_23, beta2[j])
            result += (g0 * dg1 * g2)

        dpsi.append(result)     
    return dpsi

def dpsi_over_dbeta2(centerpos1: list, centerpos2: float, delta_a_j: float,
                     L_01: float, L_12: float, L_23: float, beta1: list,
                     beta2: float):
    # Equation 24 in report
    """
    centerpos1: Slit position for first layer
    centerpos2: Slit position effected by calculated beta 
    delta_a_j: Size of a_i
    L_01: Distance between input and first layer
    L_12: Distance between first and second layer
    L_23: Distance between second and third layer
    beta1: Widths in first layer slits
    beta2: Calculated width in second layer
    """   
    a_j = [i * delta_a_j  for i in range(-500, 501)]
    dpsi = []
    
    for a in a_j:
        result = 0
        dg2 = dfunction_for_B1_or_B2(centerpos2, a, L_23, beta2)
        for j, pos1 in enumerate(centerpos1):
            centerpos1[j] = round_number(pos1)
            g0 = g_zero(centerpos1[j], L_01)
            g1 = g_one_or_g_two(centerpos1[j], centerpos2, L_12, beta1[j])
            result += (g0 * g1)

        dpsi.append(dg2 * result)     
    return dpsi

def dy_pred_over_dbeta(psi_N_conj, func_dpsi_over_dbeta, centerpos1, centerpos2,
                       delta_a_j: float, L_01: float, L_12: float, L_23: float,
                       beta1, beta2):
    # Equation 13 in report
    """
    psi_N_conj: Conjugate of psi equation 
    func_dpsi_over_dbeta: Derivative function of beta with respect to psi
    centerpos1: Slit position for first layer
    centerpos2: Slit position for second layer
    delta_a_j: Size of a_i
    L_01: Distance between input and first layer
    L_12: Distance between first and second layer
    L_23: Distance between second and third layer
    beta1: Widths in first layer slits
    beta2: Widths in second layer slits
    """   
    dpsi_N = func_dpsi_over_dbeta(centerpos1, centerpos2, delta_a_j, L_01, L_12,
                                  L_23, beta1,beta2)
    result = 2 * (dpsi_N * psi_N_conj)
    return result.real
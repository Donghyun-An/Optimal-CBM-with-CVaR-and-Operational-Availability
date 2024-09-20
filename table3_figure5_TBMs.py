import concurrent.futures
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from final_functions import *
import copy
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


def parallel_calculator(delta_range):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculator, set_risk_free, set_L_failure, set_inc_x,
                                   set_gamma_alpha, set_gamma_beta,
                                   set_ci, set_cp, set_cc, set_cd,
                                   delta, set_CVaR_rho, Avail_target,
                                   epsilon_lag, epsilon_lambda) for delta in delta_range]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
    return results

def calculator(set_risk_free, set_L_failure, set_inc_x,
               set_gamma_alpha, set_gamma_beta,
               set_ci, set_cp, set_cc, set_cd,
               set_delta, set_CVaR_rho, Avail_target,
               epsilon_lag, epsilon_lambda):

    func = functions(set_risk_free, set_L_failure, set_inc_x,
                     set_gamma_alpha, set_gamma_beta,
                     set_ci, set_cp, set_cc, set_cd,
                     set_delta, set_CVaR_rho, Avail_target,
                     epsilon_lag, epsilon_lambda)
    set_mean, set_variance = set_gamma_alpha / set_gamma_beta, set_gamma_alpha / set_gamma_beta / set_gamma_beta

    # TC, MC 값 matrix 생성
    tmp_TC_x_num_dict = {}  # np.zeros((func.x_num_Max + 1))
    tmp_Lag_TC_x_num_dict = {}  # np.zeros((func.x_num_Max + 1))
    TC_x_num_dict = {}  # np.zeros((func.x_num_Max + 1))
    Lag_TC_x_num_dict = {}
    MC_x_num_dict = {}
    Avail_xi_num_dict = {}

    # Decision 넣을 matrix 생성
    pi_x_num_dict = {}  # np.zeros((func.x_num_Max + 1))
    solution_dict = {}
    ###############################################
    # D_kx 계산 / 기존 데이터 있으면 그대로 불러옴
    D_file_path = './D_kx/Dkx_mean_variance_rho_riskfree_L_incx_delta_{}_{}_{}_{}_{}_{}_{}.csv'.format(
        set_mean, set_variance, set_CVaR_rho, set_risk_free, set_L_failure, set_inc_x, set_delta)
    if os.path.isfile(D_file_path) and not any(
            math.isnan(value) for value in func.read_csv_to_dict(D_file_path).values()):  # False:
        D_x_num_dict = func.read_csv_to_dict(D_file_path)
    else:
        start_time = time.time()
        D_x_num_dict = {}
        D_x_num_dict[(0)] = np.nan
        while any(math.isnan(value) for value in D_x_num_dict.values()):
            for x_num in np.arange(0, func.x_num_Max + 1, 1):
                result = func.CVaR_D_x_num(x_num)
                D_x_num_dict[(x_num)] = result[0]
            end_time = time.time()
            execution_time = end_time - start_time
            seconds = int(execution_time)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            print("D_x_num_dict 실행 시간: {} hours {} minutes {} seconds".format(hours, minutes, seconds))
            func.write_dict_to_csv(D_file_path, D_x_num_dict)
            print()
    ###############################################
    # T_kx 계산 / 기존 데이터 있으면 그대로 불러옴
    T_file_path = './T_kx/Tkx_mean_variance_rho_L_incx_delta_{}_{}_{}_{}_{}_{}.csv'.format(
        set_mean, set_variance, set_CVaR_rho, set_L_failure, set_inc_x, set_delta)
    if os.path.isfile(T_file_path) and not any(
            math.isnan(value) for value in func.read_csv_to_dict(T_file_path).values()):  # False:
        T_x_num_dict = func.read_csv_to_dict(T_file_path)
    else:
        start_time = time.time()
        T_x_num_dict = {}
        T_x_num_dict[(0)] = np.nan
        while any(math.isnan(value) for value in T_x_num_dict.values()):
            for x_num in np.arange(0, func.x_num_Max + 1, 1):
                result = func.T_x_num(x_num)
                T_x_num_dict[(x_num)] = result[0]
            end_time = time.time()
            execution_time = end_time - start_time
            seconds = int(execution_time)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            print("T_x_num_dict 실행 시간: {} hours {} minutes {} seconds".format(hours, minutes, seconds))
            func.write_dict_to_csv(T_file_path, T_x_num_dict)
            print()
    ###############################################
    # Avail 계산
    P_transition = func.P_transition()
    for x_num in np.arange(0, func.x_num_Max + 1, 1):
        tmp_P = func.P_steady_xinP(x_num, P_transition)
        tmp_P_steady = tmp_P[0]
        tmp_P_action = tmp_P[1]
        tmp_P_steady_after_action = np.dot(tmp_P_steady, tmp_P_action)
        Avail_xi_num_dict[(x_num)] = 1 - np.sum(
            np.array(list(T_x_num_dict.values())) * tmp_P_steady_after_action) / func.delta
    try:
        con_xi_num = max(xi for xi, avail in Avail_xi_num_dict.items() if avail >= func.Avail_target)
    except:
        con_xi_num = 0
    Avail_E = Avail_xi_num_dict[(con_xi_num)]

    tmp_P = func.P_steady_xinP_TBM(P_transition)
    tmp_P_steady = tmp_P[0]
    tmp_P_action = tmp_P[1]
    tmp_P_steady_after_action = np.dot(tmp_P_steady, tmp_P_action)
    TBM_avail = 1 - np.sum(np.array(list(T_x_num_dict.values())) * tmp_P_steady_after_action) / func.delta

    tmp_P = func.P_steady_xinP_TBM(P_transition)
    tmp_P_steady = tmp_P[0]
    tmp_P_action = tmp_P[1]
    tmp_P_steady_after_action = np.dot(tmp_P_steady, tmp_P_action)
    # print(tmp_P_steady_after_action)
    TBM_avail = 1 - np.sum(np.array(list(T_x_num_dict.values())) * tmp_P_steady_after_action) / func.delta
    # print(T_x_num_dict)
    NPV_total_cost = func.sum_ci(set_delta, set_risk_free, set_cc+D_x_num_dict[(0)] * set_cd)
    Operational_avail = TBM_avail

    return [set_delta, NPV_total_cost, Operational_avail]
    # return [TC_x_num_dict, pi_x_num_dict, sum_ci, n_iter, list_changes, xi_num, MC_x_num_dict]

set_inc_delta = 1.0
set_inc_x = 0.05
set_L_failure = 1
epsilon_lag = 2.5
epsilon_lambda = 0.1
sen_1, sen_2, sen_3 = 1, 1, 1

set_ci = 1
set_cp, set_cc, set_cd = 5 * sen_1, 12 * sen_2, 10 * sen_3
set_gamma_alpha, set_gamma_beta = 1.5, 20
set_risk_free = 0.15 / 365 / 24
set_CVaR_rho = 0
Avail_target = 0.999
set_mean, set_variance = set_gamma_alpha / set_gamma_beta, set_gamma_alpha / set_gamma_beta / set_gamma_beta
delta_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8]

file_name = '{}'.format('tbm_xo_xx')

para_dict = {'riskfree': set_risk_free,
             'L_failure': set_L_failure, 'inc_x': set_inc_x,
             'alpha': set_gamma_alpha, 'beta': set_gamma_beta, 'mean': set_mean, 'variance': set_variance,
             'ci': set_ci, 'cp': set_cp, 'cc': set_cc, 'cd': set_cd,
             'CVaR_rho': set_CVaR_rho, 'Avail_target': Avail_target,
             'epsilon_lag': epsilon_lag, 'epsilon_lambda': epsilon_lambda}
para_excel = pd.DataFrame(list(para_dict.items()), columns=['name', 'value'])
print(file_name)
print(para_dict)

results = parallel_calculator(delta_range)

print(results)
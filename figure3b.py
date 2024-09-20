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
    tmp_func = functions(set_risk_free, set_L_failure, set_inc_x,
                         set_gamma_alpha, set_gamma_beta,
                         set_ci, set_cp, set_cc, set_cd,
                         set_delta, 0, Avail_target,
                         epsilon_lag, epsilon_lambda)

    func = functions(set_risk_free, set_L_failure, set_inc_x,
                     set_gamma_alpha, set_gamma_beta,
                     set_ci, set_cp, set_cc, set_cd,
                     set_delta, set_CVaR_rho, Avail_target,
                     epsilon_lag, epsilon_lambda)
    set_mean, set_variance = set_gamma_alpha / set_gamma_beta, set_gamma_alpha / set_gamma_beta / set_gamma_beta

    def cal_TC_dict_MC_dict_xi_num(TC_x_num_dict, MC_x_num_dict, xi_num, Avail_E, lag_lambda, epsilon_lag):
        new_Lag_x_num_dict = {}
        new_TC_x_num_dict = {}
        new_MC_x_num_dict = {}
        change = epsilon_lag
        previous_TC = TC_x_num_dict
        previous_MC = MC_x_num_dict
        while change >= epsilon_lag:
            new_TC_x_num_dict[(0)] = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(0,
                                                                                                   previous_TC) + \
                                     D_x_num_dict[(0)] * set_cd
            new_MC_x_num_dict[0] = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(0, previous_MC)

            for x_num in np.arange(1, func.x_num_Max + 1, 1):
                if x_num * func.inc_x >= func.L_failure:
                    new_TC_x_num_dict[(x_num)] = func.cost_corrective + new_TC_x_num_dict[(0)]
                    new_MC_x_num_dict[(x_num)] = func.cost_corrective + new_MC_x_num_dict[(0)]
                elif x_num > xi_num:
                    new_TC_x_num_dict[(x_num)] = func.cost_preventive + new_TC_x_num_dict[(0)]
                    new_MC_x_num_dict[(x_num)] = func.cost_preventive + new_MC_x_num_dict[(0)]

                else:
                    new_TC_x_num_dict[(x_num)] = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(x_num,
                                                                                                               TC_x_num_dict) + \
                                                 D_x_num_dict[(x_num)] * set_cd
                    new_MC_x_num_dict[(x_num)] = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(x_num,
                                                                                                               previous_MC)
            change = func.max_dict_value_Rdiff_dictdict(previous_TC, new_TC_x_num_dict)
            previous_TC = new_TC_x_num_dict
            previous_MC = new_MC_x_num_dict

        for x_num in np.arange(0, func.x_num_Max + 1, 1):
            new_Lag_x_num_dict[(x_num)] = new_TC_x_num_dict[(x_num)] + lag_lambda * (func.Avail_target - Avail_E)

        return new_TC_x_num_dict, new_MC_x_num_dict, new_Lag_x_num_dict

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

    ###############################################
    # Initialize TC_x(x), MC_x(x) lambda_x
    for x_num in np.arange(0, func.x_num_Max + 1, 1):
        tmp_TC_x_num_dict[(x_num)] = 0
        tmp_Lag_TC_x_num_dict[(x_num)] = 0

        TC_x_num_dict[(x_num)] = 0
        Lag_TC_x_num_dict[(x_num)] = 0
        MC_x_num_dict[(x_num)] = 0
        if x_num == func.x_num_Max:
            pi_x_num_dict[(x_num)] = 2
        elif x_num > con_xi_num:
            pi_x_num_dict[(x_num)] = 1
        else:
            pi_x_num_dict[(x_num)] = 0
    lag_lambda = 0
    xi_num = con_xi_num  # func.x_num_Max

    # list_lag_changes = []
    # list_lambda_changes = []

    change_lag = epsilon_lag
    change_lambda = epsilon_lambda
    lambda_iter = 0
    lag_iter = 0

    sum_ci = func.sum_ci(set_delta, set_risk_free, set_ci)
    ###############################################
    if max(Avail_xi_num_dict.values()) < func.Avail_target:
        return [TC_x_num_dict, MC_x_num_dict, sum_ci, pi_x_num_dict,
                func.P_steady_xinP(0, P_transition),
                set_delta, 0, Avail_xi_num_dict[(0)], 0, 0]
    ###############################################
    while change_lambda >= epsilon_lambda or len(solution_dict) < 1 or lambda_iter < 14:
        # Check feasibility and update policy set
        if Avail_E >= func.Avail_target:
            solution_dict_key = tuple(pi_x_num_dict.items())
            solution_dict[solution_dict_key] = [xi_num, pi_x_num_dict,
                                                func.Avail_E_piPtT(pi_x_num_dict, P_transition, T_x_num_dict),
                                                P_transition]
            print("Obj, Lag_lambda: {}, {}".format(
                np.sum(np.array(list(TC_x_num_dict.values())) * func.P_steady_piP(pi_x_num_dict, P_transition)[0]),
                lag_lambda))  # Lag_TC_x_num_dict[0]))
        print('------------------------------')

        while change_lag >= epsilon_lag:
            tmp_previous_TC_x_num_dict = copy.deepcopy(tmp_TC_x_num_dict)
            tmp_previous_Lag_TC_x_num_dict = copy.deepcopy(tmp_Lag_TC_x_num_dict)

            previous_TC_x_num_dict = copy.deepcopy(TC_x_num_dict)
            previous_Lag_TC_x_num_dict = copy.deepcopy(Lag_TC_x_num_dict)
            previous_MC_x_num_dict = copy.deepcopy(MC_x_num_dict)
            previous_Avail_E = copy.deepcopy(Avail_E)
            # previous_xi_num = copy.deepcopy(xi_num)
            previous_change_lag = change_lag

            # if x_num = 0 then DN
            tmp_TC_x_num_dict[(0)] = np.exp(-tmp_func.risk_free_rate * tmp_func.delta) * tmp_func.CVaR_C_x_num(0,
                                                                                                               tmp_previous_TC_x_num_dict) + \
                                     D_x_num_dict[(0)] * set_cd
            tmp_Lag_TC_x_num_dict[(0)] = np.exp(-tmp_func.risk_free_rate * tmp_func.delta) * tmp_func.CVaR_C_x_num(0,
                                                                                                                   tmp_previous_Lag_TC_x_num_dict) + \
                                         D_x_num_dict[(0)] * set_cd + lag_lambda * (
                                                 tmp_func.Avail_target - previous_Avail_E)

            TC_x_num_dict[(0)] = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(0,
                                                                                               previous_TC_x_num_dict) + \
                                 D_x_num_dict[(0)] * set_cd
            Lag_TC_x_num_dict[(0)] = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(0,
                                                                                                   previous_Lag_TC_x_num_dict) + \
                                     D_x_num_dict[(0)] * set_cd + lag_lambda * (func.Avail_target - previous_Avail_E)
            # (func.Avail_target - previous_Avail_E)
            MC_x_num_dict[0] = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(0, previous_MC_x_num_dict)
            pi_x_num_dict[(0)] = 0
            xi_num = 0

            for x_num in np.arange(1, func.x_num_Max + 1, 1):
                if x_num * func.inc_x >= func.L_failure:
                    tmp_TC_x_num_dict[(x_num)] = func.cost_corrective + tmp_TC_x_num_dict[(0)]
                    tmp_Lag_TC_x_num_dict[(x_num)] = tmp_func.cost_corrective + tmp_Lag_TC_x_num_dict[
                        (0)] + lag_lambda * (
                                                             tmp_func.Avail_target - tmp_func.Avail_E_piPtT(
                                                         pi_x_num_dict, P_transition, T_x_num_dict))

                    TC_x_num_dict[(x_num)] = func.cost_corrective + TC_x_num_dict[(0)]
                    Lag_TC_x_num_dict[(x_num)] = func.cost_corrective + Lag_TC_x_num_dict[(0)] + lag_lambda * (
                            func.Avail_target - func.Avail_E_piPtT(pi_x_num_dict, P_transition, T_x_num_dict))
                    MC_x_num_dict[(x_num)] = func.cost_corrective + MC_x_num_dict[(0)]
                    pi_x_num_dict[(x_num)] = 2

                else:
                    # x_num까지만 DN
                    tmp_PM = tmp_func.cost_preventive + tmp_TC_x_num_dict[(0)]
                    tmp_DN = np.exp(-tmp_func.risk_free_rate * tmp_func.delta) * tmp_func.CVaR_C_x_num(x_num,
                                                                                                       tmp_previous_TC_x_num_dict) + \
                             D_x_num_dict[(x_num)] * set_cd

                    PM = func.cost_preventive + TC_x_num_dict[(0)]
                    DN = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(x_num, previous_TC_x_num_dict) + \
                         D_x_num_dict[(x_num)] * set_cd

                    tmp_PM_pi_x_num_dict = copy.deepcopy(pi_x_num_dict)
                    tmp_PM_pi_x_num_dict[(x_num)] = 1
                    tmp_PM_Avail = func.Avail_E_piPtT(tmp_PM_pi_x_num_dict, P_transition, T_x_num_dict)

                    tmp_Lag_PM = tmp_func.cost_preventive + tmp_Lag_TC_x_num_dict[(0)] + lag_lambda * (
                            tmp_func.Avail_target - tmp_PM_Avail)
                    Lag_PM = func.cost_preventive + Lag_TC_x_num_dict[(0)] + lag_lambda * (
                            func.Avail_target - tmp_PM_Avail)

                    tmp_DN_pi_x_num_dict = copy.deepcopy(pi_x_num_dict)
                    tmp_DN_pi_x_num_dict[(x_num)] = 0
                    tmp_DN_Avail = func.Avail_E_piPtT(tmp_DN_pi_x_num_dict, P_transition, T_x_num_dict)
                    tmp_Lag_DN = np.exp(-tmp_func.risk_free_rate * tmp_func.delta) * tmp_func.CVaR_C_x_num(x_num,
                                                                                                           tmp_previous_Lag_TC_x_num_dict) + \
                                 D_x_num_dict[(x_num)] * set_cd + lag_lambda * (tmp_func.Avail_target - tmp_DN_Avail)
                    Lag_DN = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(x_num,
                                                                                           previous_Lag_TC_x_num_dict) + \
                             D_x_num_dict[(x_num)] * set_cd + lag_lambda * (func.Avail_target - tmp_DN_Avail)

                    if tmp_Lag_PM <= tmp_Lag_DN:
                        tmp_TC_x_num_dict[(x_num)] = tmp_PM
                        tmp_Lag_TC_x_num_dict[(x_num)] = tmp_Lag_PM

                        TC_x_num_dict[(x_num)] = PM
                        Lag_TC_x_num_dict[(x_num)] = Lag_PM
                        MC_x_num_dict[(x_num)] = func.cost_preventive + MC_x_num_dict[(0)]
                        pi_x_num_dict[(x_num)] = 1

                    else:
                        tmp_TC_x_num_dict[(x_num)] = tmp_DN
                        tmp_Lag_TC_x_num_dict[(x_num)] = tmp_Lag_DN

                        TC_x_num_dict[(x_num)] = DN
                        Lag_TC_x_num_dict[(x_num)] = Lag_DN
                        MC_x_num_dict[(x_num)] = np.exp(-func.risk_free_rate * func.delta) * func.CVaR_C_x_num(x_num,
                                                                                                               previous_MC_x_num_dict)
                        pi_x_num_dict[(x_num)] = 0
                        xi_num = x_num  # 같으면 DN, DN 중 가장 큰게 xi

            Avail_E = func.Avail_E_piPtT(pi_x_num_dict, P_transition, T_x_num_dict)
            # try:
            #     xi_num = min(key for key, value in pi_x_num_dict.items() if value == 1) - 1
            # except:
            #     xi_num = 0

            # print(Dmat_CVaR_TC_x_num)
            change_lag = func.max_dict_value_diff_dictdict(previous_Lag_TC_x_num_dict, Lag_TC_x_num_dict)
            # list_lag_changes.append(change_lag)

            # 연속된 숫자 체크
            if change_lag == previous_change_lag:
                count += 1
            else:
                count = 0

            if count == 50000:
                break

            # if lag_iter % 500 == 0:
            print()
            print("lambda_iter, lag_iter: {}, {}".format(lambda_iter, lag_iter))
            print("lambda_change, lag_change: {}, {}".format(change_lambda, change_lag))
            print("con_xi, xi, Avail: {}, {}, {}".format(con_xi_num, xi_num, Avail_E))
            print('pi: {}'.format(pi_x_num_dict))
            print('lag_lambda: {}'.format(lag_lambda))
            print("num_solution: {}".format(len(solution_dict)), solution_dict.keys())
            lag_iter = lag_iter + 1

        previous_lag_lambda = copy.deepcopy(lag_lambda)
        lag_lambda = max(0, lag_lambda + func.eta(lambda_iter) * (func.Avail_target - Avail_E))
        # change_lambda = abs(previous_lag_lambda - lag_lambda)
        # change_lambda = func.max_dict_value_diff_dictdict(previous_lambda_dict, lambda_dict)
        if previous_lag_lambda != 0:
            change_lambda = abs((previous_lag_lambda - lag_lambda) / previous_lag_lambda)
            # change_lambda = epsilon_lambda #abs(previous_lag_lambda - lag_lambda)
        else:
            # change_lambda = epsilon_lambda
            if lag_lambda != 0:
                change_lambda = epsilon_lambda
            else:
                change_lambda = 0
        # list_lambda_changes.append(change_lambda)
        lambda_iter = lambda_iter + 1
        change_lag = epsilon_lag  # change_lag 초기화
        print()
        print("lambda_iter, lag_iter: {}, {}".format(lambda_iter, lag_iter))
        print("lambda_change, lag_change: {}, {}".format(change_lambda, change_lag))
        print("con_xi, xi, Avail: {}, {}, {}".format(con_xi_num, xi_num, Avail_E))
        print('pi: {}'.format(pi_x_num_dict))
        print('lag_lambda: {}, lag_delta: {}'.format(lag_lambda, func.eta(lambda_iter) * (func.Avail_target - Avail_E)))
        print("num_solution: {}".format(len(solution_dict)))

    # 솔루션 다 구하고 나머지 상태
    for key, values in solution_dict.items():
        tmp_TC_MC = cal_TC_dict_MC_dict_xi_num(TC_x_num_dict, MC_x_num_dict, values[0], values[2], lag_lambda,
                                               epsilon_lag)
        tmp_TC_dict = tmp_TC_MC[0]
        tmp_MC_dict = tmp_TC_MC[1]
        tmp_Lag_dict = tmp_TC_MC[2]

        solution_dict[key].insert(0, tmp_Lag_dict)
        solution_dict[key].insert(1, tmp_TC_dict)
        solution_dict[key].insert(2, tmp_MC_dict)

    opt_key = min(solution_dict, key=lambda pi_key: np.sum(
        np.array(list(solution_dict[pi_key][1].values())) *
        func.P_steady_piP(solution_dict[pi_key][4], solution_dict[pi_key][6])[0]))
    opt_Lag_TC_dict = solution_dict[opt_key][0]
    opt_TC_dict = solution_dict[opt_key][1]
    opt_MC_dict = solution_dict[opt_key][2]
    opt_xi_num = solution_dict[opt_key][3]
    opt_pi_dict = solution_dict[opt_key][4]
    opt_Avail = solution_dict[opt_key][5]
    opt_lambda = lag_lambda

    return [opt_TC_dict, opt_MC_dict, sum_ci, opt_pi_dict,
            func.P_steady_piP(pi_x_num_dict, P_transition),
            set_delta, opt_lambda, opt_Avail, con_xi_num, opt_xi_num]
    # return [TC_x_num_dict, pi_x_num_dict, sum_ci, n_iter, list_changes, xi_num, MC_x_num_dict]

list_delta = []
list_lag_lambda = []
list_avail = []
list_con_xi_num = []
list_xi_num = []

NPV_inspection = []
NPV_downtime = []
NPV_maintenance = []
NPV_maintenance_and_downtime = []
NPV_all = []
set_inc_delta = 1.0
set_inc_x = 0.05
set_L_failure = 1
epsilon_lag = 2.5
epsilon_lambda = 0.1
sen_1, sen_2, sen_3 = 1, 1, 1

for set_CVaR_rho in [0, 0.25, 0.75, 0.99]:

    set_ci = 1
    set_cp, set_cc, set_cd = 5 * sen_1, 12 * sen_2, 10 * sen_3
    set_gamma_alpha, set_gamma_beta = 1.5, 20
    set_risk_free = 0.15 / 365 / 24
    # set_CVaR_rho = 0.5
    Avail_target = 0.999
    set_mean, set_variance = set_gamma_alpha / set_gamma_beta, set_gamma_alpha / set_gamma_beta / set_gamma_beta
    delta_range = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8]

    file_name = '{}_{}'.format('sen_rho_', set_CVaR_rho)
    writer = pd.ExcelWriter('./result/{}.xlsx'.format(file_name),
                            engine='xlsxwriter')

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

    list_columns = [''] + [x for x in np.arange(int(set_L_failure / set_inc_x) + 1)]
    V_maintenance_and_downtime_excel = pd.DataFrame(columns=list_columns)
    V_maintenance_excel = pd.DataFrame(columns=list_columns)
    V_downtime_excel = pd.DataFrame(columns=list_columns)
    pi_excel = pd.DataFrame(columns=list_columns)
    # Now you can process the results as needed
    start_time = time.time()
    for a in results:
        # Process the result for each delta

        # print(f"Results for delta={a[5]}: {a}")

        list_delta.append(a[5])
        list_lag_lambda.append(a[6])
        list_avail.append(a[7])
        list_con_xi_num.append(a[8])
        list_xi_num.append(a[9])

        NPV_maintenance_and_downtime.append(np.sum(np.array(list(a[0].values())) * a[4][0]))

        V_maintenance_and_downtime_excel = V_maintenance_and_downtime_excel.append(
            pd.DataFrame(np.append(a[5], list(a[0].values())).reshape(1, -1), columns=list_columns),
            ignore_index=True)
        V_maintenance_excel = V_maintenance_excel.append(
            pd.DataFrame(np.append(a[5], list(a[1].values())).reshape(1, -1), columns=list_columns),
            ignore_index=True)
        V_downtime_excel = V_downtime_excel.append(
            pd.DataFrame(np.append(a[5], np.subtract(list(a[0].values()), list(a[1].values()))).reshape(1, -1),
                         columns=list_columns),
            ignore_index=True)
        pi_excel = pi_excel.append(pd.DataFrame(np.append(a[5], list(a[3].values())).reshape(1, -1), columns=list_columns),
                                   ignore_index=True)

        NPV_inspection.append(a[2])
        NPV_downtime.append(
            np.sum(np.array(list(a[0].values())) * a[4][0]) - np.sum(np.array(list(a[1].values())) * a[4][0]))
        NPV_maintenance.append(np.sum(np.array(list(a[1].values())) * a[4][0]))
        # NPV_downtime.append(a[1][0] - a[2][0])
        # NPV_maintenance.append(a[2][0])
        NPV_all.append(np.sum(np.array(list(a[0].values())) * a[4][0]) + a[2])

        # print('====================')
        result = pd.DataFrame(
            list(zip(list_delta, list_lag_lambda, list_avail, list_con_xi_num, list_xi_num,
                     NPV_all, NPV_inspection, NPV_maintenance_and_downtime, NPV_downtime, NPV_maintenance)),
            columns=['delta', 'lag_lambda', 'avail', 'con_xi_num', 'xi_num',
                     'NPV_of_total_cost', 'NPV_of_inspection_costs', 'NPV_of_maintenance_and_downtime_costs',
                     'NPV_of_downtime_costs', 'NPV_of_maintenance_costs'])
        # print(result.to_string(index=False))
        # print('====================')

    end_time = time.time()
    execution_time = end_time - start_time
    seconds = int(execution_time)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    sorted_result = result.sort_values(by='delta')
    sorted_result.reset_index(drop=True).to_excel(writer, sheet_name='Costs')

    print(sorted_result.to_string(index=False))
    print("{} 실행 시간: {} hours {} minutes {} seconds".format(a[6], hours, minutes, seconds))
    print(file_name)
    print(para_dict)

    pi_excel.T.iloc[::-1].reset_index(drop=True).sort_values(by=len(pi_excel.columns) - 1, axis=1).to_excel(writer,
                                                                                                            sheet_name='Policy')
    V_maintenance_and_downtime_excel.T.iloc[::-1].reset_index(drop=True).sort_values(by=len(pi_excel.columns) - 1,
                                                                                     axis=1).to_excel(writer,
                                                                                                      sheet_name='V_m_d')
    V_maintenance_excel.T.iloc[::-1].reset_index(drop=True).sort_values(by=len(pi_excel.columns) - 1, axis=1).to_excel(
        writer, sheet_name='V_m')
    V_downtime_excel.T.iloc[::-1].reset_index(drop=True).sort_values(by=len(pi_excel.columns) - 1, axis=1).to_excel(writer,
                                                                                                                    sheet_name='V_d')
    para_excel.to_excel(writer, sheet_name='para')
    writer.save()
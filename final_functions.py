# wts 일단 수정
import scipy.integrate
from scipy.integrate import quad
from sympy import *
import math
import numpy as np
from scipy import stats
import time
import pandas as pd
import csv
import ast
import mpmath
# import sympy

class functions:
    def __init__(self, risk_free_rate, L_failure, inc_x, gamma_alpha, gamma_beta, cost_inspection,
                 cost_preventive, cost_corrective, cost_downtime, delta, CVaR_rho, Avail_target,
               epsilon_lag, epsilon_lambda):

        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta

        self.cost_inspection = cost_inspection
        self.cost_preventive = cost_preventive
        self.cost_corrective = cost_corrective
        self.cost_downtime = cost_downtime

        self.L_failure = L_failure
        self.risk_free_rate = risk_free_rate

        self.inc_x = inc_x

        self.delta = delta

        self.CVaR_rho = CVaR_rho
        self.Avail_target = Avail_target

        self.x_num_Max = int(self.L_failure / self.inc_x)

        self.epsilon_lag = epsilon_lag
        self.epsilon_lambda = epsilon_lambda

    ###################################################################################################
    def p_xn_xn_prime(self, x_n, x_n_prime):
        # 내 state 변화
        if x_n_prime == self.x_num_Max:
            tmp_p = 1 - stats.gamma.cdf((x_n_prime - x_n) * self.inc_x, self.gamma_alpha * self.delta, 0,
                                        1 / self.gamma_beta)
        elif x_n_prime >= x_n:
            tmp_p = stats.gamma.cdf((x_n_prime - x_n + 1) * self.inc_x, self.gamma_alpha * self.delta, 0,
                                    1 / self.gamma_beta) - stats.gamma.cdf((x_n_prime - x_n) * self.inc_x,
                                                                           self.gamma_alpha * self.delta, 0,
                                                                           1 / self.gamma_beta)
        else:
            tmp_p = 0
        return tmp_p

    ###################################################################################################
    def CVaR_D_x_num(self, x_num):

        time_start = time.time()
        print('\t D x_num: {}'.format(x_num))
        print('\t start: {}'.format(time.strftime('%H:%M:%S', time.localtime(time_start))))
        sym_alpha = Symbol('sym_alpha', real=True)
        sym_beta = Symbol('sym_beta', real=True)

        sym_r = Symbol('sym_r', real=True, positive=True)

        sym_tf = Symbol('sym_tf', real=True, domain=Interval(0, self.delta))
        sym_tr = Symbol('sym_tr', real=True, positive=True, domain=Interval(0, oo))

        sym_x = Symbol('sym_x', real=True, domain=Interval(self.L_failure, oo))

        sym_L = Symbol('sym_L', real=True, positive=True)
        if x_num < self.x_num_Max:
            # tf에 고장날 확률
            cdf_f = 1 - integrate(exp(-sym_x * sym_beta) * (sym_beta ** (sym_alpha * sym_tf)) * (
                    sym_x ** (sym_alpha * sym_tf - 1)) / gamma(
                sym_alpha * sym_tf), (sym_x, 0, sym_L))
            pdf_f = diff(cdf_f, sym_tf)

            # tf에 고장난 평균 비용
            # sym_d_t = integrate(E**(-sym_r * sym_tmp), (sym_tmp, sym_tf, sym_tr))
            sym_d_t = - exp(-sym_r * sym_tr) / sym_r + exp(-sym_r * sym_tf) / sym_r

            # tf에 고장난 평균 비용
            # print('ed')
            ed = sym_d_t * pdf_f
            # print(ed)

            # print('sub_ed')
            sub_ed = ed.evalf(subs={sym_alpha: self.gamma_alpha, sym_beta: self.gamma_beta, sym_tr: self.delta,
                                    sym_L: self.L_failure - x_num * self.inc_x, sym_r: self.risk_free_rate})
            # print(sub_ed)

            mpmath.mp.dps = 200  # 소수점 이하 100자리까지 설정
            # mpmath.mp.maxprec = 5000  # 최대 허용 정밀도

            sol = quad(lambdify(sym_tf, sub_ed, modules=['sympy', 'numpy']), 0, self.delta)[0]
        else:
            # tf에 고장난 평균 비용
            # sym_d_t = integrate(E**(-sym_r * sym_tmp), (sym_tmp, sym_tf, sym_tr))
            sol = - exp(- self.risk_free_rate * self.delta) / self.risk_free_rate + 1 / self.risk_free_rate

        # print('\t \t CVaR_D: {}'.format(sol))
        time_end = time.time()
        print('\t end: {}'.format(time.strftime('%H:%M:%S', time.localtime(time_end))))
        print('\t running time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time_end - time_start))))
        print()
        return [sol]

    def CVaR_C_x_num(self, x_num, C_dict):
        value = 0
        p_sum = 0
        is_first = 0

        list_prob = []
        list_value = []

        for x_num_prime in np.arange(x_num, self.x_num_Max + 1, 1):
            list_prob.append(self.p_xn_xn_prime(x_num, x_num_prime))
            list_value.append(C_dict[(x_num_prime)])

        # list_value, list_prob = (list(t) for t in zip(*sorted(zip(list_value, list_prob))))

        if list_prob[-1] > 1 - self.CVaR_rho:
            # print('else')
            return list_value[-1]
        else:
            for v, p in zip(list_value, list_prob):
                p_sum = p_sum + p
                if p_sum >= self.CVaR_rho:
                    if is_first == 0:
                        value = value + (p_sum - self.CVaR_rho) * v
                        is_first = is_first + 1
                    else:
                        value = value + p * v

            return value / (1 - self.CVaR_rho)

    def T_x_num(self, x_num):
        time_start = time.time()
        if x_num < self.x_num_Max:
            print('\t T  x_num: {}'.format(x_num))
            print('\t start: {}'.format(time.strftime('%H:%M:%S', time.localtime(time_start))))

            sym_alpha = Symbol('sym_alpha', real=True)
            sym_beta = Symbol('sym_beta', real=True)

            sym_tf = Symbol('sym_tf', real=True, domain=Interval(0, self.delta))
            sym_tr = Symbol('sym_tr', real=True, positive=True, domain=Interval(0, oo))

            sym_x = Symbol('sym_x', real=True, domain=Interval(self.L_failure, oo))

            sym_L = Symbol('sym_L', real=True, positive=True)

            # tf에 고장날 확률
            cdf_f = 1 - integrate(exp(-sym_x * sym_beta) * (sym_beta ** (sym_alpha * sym_tf)) * (
                    sym_x ** (sym_alpha * sym_tf - 1)) / gamma(
                sym_alpha * sym_tf), (sym_x, 0, sym_L))
            pdf_f = diff(cdf_f, sym_tf)

            # tf에 고장난 평균 시간
            et = (sym_tr - sym_tf) * pdf_f

            # print('sub_ed')
            sub_et = et.evalf(subs={sym_alpha: self.gamma_alpha, sym_beta: self.gamma_beta, sym_tr: self.delta,
                                    sym_L: self.L_failure - x_num * self.inc_x})
            # print(sub_ed)

            sol = quad(lambdify(sym_tf, sub_et, modules=['sympy', 'numpy']), 0, self.delta)[0]
        else:
            sol = self.delta

        # print('\t \t CVaR_D: {}'.format(sol))
        time_end = time.time()
        print('\t end: {}'.format(time.strftime('%H:%M:%S', time.localtime(time_end))))
        print('\t running time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time_end - time_start))))
        print()
        return [sol]

    def CVaR_steady(self, list_prob, list_value):
        value = 0
        p_sum = 0
        is_first = 0
        if list_prob[-1] > 1 - self.CVaR_rho:
            # print('else')
            return list_value[-1]
        else:
            for v, p in zip(list_value, list_prob):
                p_sum = p_sum + p
                if p_sum >= self.CVaR_rho:
                    if is_first == 0:
                        value = value + (p_sum - self.CVaR_rho) * v
                        is_first = is_first + 1
                    else:
                        value = value + p * v
            return value / (1 - self.CVaR_rho)

    def sum_ci(self, delta, risk_free_rate, ci):
        return ci / (1 - np.exp(-risk_free_rate * delta))

    def select_and_move_to_last(self, arr, n):
        arr_out = np.zeros_like(arr)
        arr_out[-1] = arr[n]
        return arr_out

    def max_dict_value_diff_dictdict(self, dict1, dict2):
        max_diff = 0
        for key in dict1.keys():
            diff = abs(dict1[key] - dict2[key])
            max_diff = max(max_diff, diff)
        return max_diff

    def max_dict_value_Rdiff_dictdict(self, dict1, dict2):
        max_val = abs(max(dict1.values()))
        if max_val == 0:
            return self.epsilon_lag
        max_diff = 0
        for key in dict1.keys():
            diff = abs(dict1[key] - dict2[key])
            max_diff = max(max_diff, diff)
        return max_diff / max_val

    def read_csv_to_dict(self, file_name):
        data_dict = {}
        with open(file_name, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                key = eval(row["Key"])
                value = eval(row["Value"])
                data_dict[key] = value
        return data_dict

    def write_dict_to_csv(self, file_name, dictionary):
        with open(file_name, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Key", "Value"])  # Write header row
            for key, value in dictionary.items():
                writer.writerow([key, value])

    def P_transition(self):
        tmp_P_transition = []
        for x_num in range(0, self.x_num_Max + 1):
            tmp_tmp_P_transition = []
            for x_num_prime in range(0, self.x_num_Max + 1):
                tmp_tmp_P_transition.append(self.p_xn_xn_prime(x_num, x_num_prime))
            tmp_P_transition.append(tmp_tmp_P_transition)
        P_transition = np.array(tmp_P_transition)
        return P_transition

    def P_steady_xinP(self, xi_n, P_transition):
        tmp_P_action = []
        for s_n in range(0, self.x_num_Max + 1):
            tmp_s_n_list = [0] * (self.x_num_Max + 1)
            if s_n > xi_n:
                tmp_s_n_list[0] = 1
            else:
                tmp_s_n_list[s_n] = 1
            tmp_P_action.append(tmp_s_n_list)
        P_action = np.array(tmp_P_action)

        P = np.dot(P_action, P_transition)

        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # Find the index of the eigenvalue 1 (assuming it exists)
        index_1 = np.where(np.isclose(eigenvalues, 1))[0][0]
        # Extract the corresponding eigenvector
        P_steady = eigenvectors[:, index_1].real.T
        # Normalize the eigenvector to ensure it's a valid solution
        P_steady /= np.sum(P_steady)
        return [P_steady, P_action]

    def P_steady_xinP_TBM(self, P_transition):
        tmp_P_action = []
        for s_n in range(0, self.x_num_Max + 1):
            tmp_s_n_list = [0] * (self.x_num_Max + 1)
            tmp_s_n_list[0] = 1
            tmp_P_action.append(tmp_s_n_list)
        P_action = np.array(tmp_P_action)

        P = np.dot(P_action, P_transition)

        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # Find the index of the eigenvalue 1 (assuming it exists)
        index_1 = np.where(np.isclose(eigenvalues, 1))[0][0]
        # Extract the corresponding eigenvector
        P_steady = eigenvectors[:, index_1].real.T
        # Normalize the eigenvector to ensure it's a valid solution
        P_steady /= np.sum(P_steady)
        return [P_steady, P_action]

    def P_steady_piP(self, pi, P_transition):
        tmp_P_action = []
        i=0
        for action in pi.values():
            tmp_s_n_list = [0] * (self.x_num_Max + 1)
            if action > 0:
                tmp_s_n_list[0] = 1
            else:
                tmp_s_n_list[i] = 1
            tmp_P_action.append(tmp_s_n_list)
            i=i+1
        P_action = np.array(tmp_P_action)

        P = np.dot(P_action, P_transition)

        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        # Find the index of the eigenvalue 1 (assuming it exists)
        index_1 = np.where(np.isclose(eigenvalues, 1))[0][0]
        # Extract the corresponding eigenvector
        P_steady = eigenvectors[:, index_1].real.T
        # Normalize the eigenvector to ensure it's a valid solution
        P_steady /= np.sum(P_steady)
        return [P_steady, P_action]

    def Avail_E_piPtT(self, pi, P_transition, T_x_num_dict):
        tmp_P = self.P_steady_piP(pi, P_transition)
        P_steady = tmp_P[0]
        P_action = tmp_P[1]
        P_steady_after_action = np.dot(P_steady, P_action)
        return 1 - np.sum(np.array(list(T_x_num_dict.values())) * P_steady_after_action) / self.delta

    def eta(self, iter):
        return 10000 / (1 + iter)
# desired data result looks like (no EV right now):
# Time(m/t), Cloud cover(o), GHI(ghi), Temperature(temp), Homeload(hl), Power generated(ac), Battery(b), weekday(w)
# datetime, EV, o, ghi, temp, hl, ac, w, b
import csv
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from sklearn import ensemble

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


class EnergyEnvironment:

    def __init__(self, mode="ground_truth", charge_mode="Austin", payment_cycle=24,datafile="test"):
        self.state = []
        self.current_index = 0
        self.sell_back = 0.0
        self.two_meter = True
        self.maximum_battery = 6.4
        self.eff_c = 0.95
        self.eff_d = 0.95
        self.eff_pleak = 0
        # self.eff_cleak = 0
        self.new_clf = None
        self.eff_cleak = 0.0001 * self.maximum_battery
        self.hour_price = list()
        self.payment_cycle = payment_cycle
        self.total_price = 0
        self.current_payment = 1
        self.month_starter = 0
        #self.price_starter = 0
        self.datafile = datafile
        self.battery_starter = (self.maximum_battery // 1) // 2
        self.max_charge_rate = 0
        # https://austinenergy.com/ae/residential/rates
        # https://www.xcelenergy.com/staticfiles/xe-responsive/Marketing/TX-Time-of-use-rate-FAQ.pdf
        # Austin or xcel or own
        self.charge_mode = charge_mode
        # print("sell back price",self.sell_back)
        # print("mamximum battery",self.maximum_battery)
        # print("charge_mode",self.charge_mode)
        # print("filename",self.datafile)
        # if mode == "ground_truth":
        #     self.init_ground_truth()
        if self.charge_mode == "own":
            self.init_price()

    def init_price(self):

        if not os.path.exists('power_price.csv'):
            print("No power_price.csv was found. Run generalpowerprice.py first.")
            raise ValueError

        with open("power_price.csv", 'r') as input_csv:

            reader = csv.reader(input_csv, delimiter=',')
            row_number = 0

            for row in reader:

                row_number += 1
                if row_number == 1 or len(row) == 0:
                    continue
                self.hour_price.append(list(map(float, row))[1:])

    def init_ground_truth(self):


        if not os.path.exists(self.datafile):
            print("No processed_hhdata_26_result.csv was found. Run generatepower.py first.")
            raise ValueError

        with open(self.datafile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            row_count = sum(1 for _ in reader)

        #with open("processed_hhdata_26_result.csv", 'r') as csvfile:
        with open(self.datafile, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            row_number = 0

            for row in reader:

                row_number += 1

                if row_number == 1:
                    continue

                # print("this is to datatime-------------------",pd.to_datetime(row[1][:-3]))

                row_data = np.array([float(row[8]),
                                     None,
                                     float(row[4]),
                                     float(row[5]),
                                     float(row[3]),
                                     float(row[2]),
                                     float(row[9]),
                                     # 0.0,
                                     float(row[6]),
                                     3,
                                     float(row[7])])

                self.state.append(row_data)

                print("\rEnvironment setup progress: %5.2f%%" % (row_number * 100 / row_count), end='')

        print("\rEnvironment setup finished. Total %i lines data." % row_count)
        #take only part of whole year for test

    # action = [charge/discharge, next_state_id]
    # check max_charge_rate, and min_charge is hl - b
    # hour, EV, o, ghi, temp, hl, ac, w, b, month
    # 0,        1,  2, 3,   4,    5,  6,  7, 8, 9

    def step(self, action):

        self.current_index += 1

        if action>0:
            deltaB = action*self.eff_c
        else:
            deltaB = action/self.eff_d

        self.state[self.current_index][8] = min(max(self.state[self.current_index - 1][8]*(1-self.eff_pleak) + deltaB - self.eff_cleak, 0), self.maximum_battery)

        #self.state[self.current_index][8] = max(action + self.state[self.current_index - 1][8], 0)


        # How much we charge the battery from grid
        charge_from_grid = max(action - self.state[self.current_index - 1][6], 0)
        total_usable_power = max(self.state[self.current_index - 1][6] - action, 0)
        total_sell_back = max(total_usable_power - self.state[self.current_index - 1][5], 0)
        total_need_grid = charge_from_grid + max(self.state[self.current_index - 1][5] - total_usable_power, 0)
        reward = 0

        if self.two_meter:
            self.total_price += -total_sell_back * self.sell_back
        else:
            total_need_grid -= total_sell_back
        if self.charge_mode == "Austin":
            self.total_price += 0.1 * total_need_grid + 0.1 * max(total_need_grid - 500, 0)

        elif self.charge_mode == "xcel":
            self.total_price += 0.056101 * total_need_grid
            if self.state[self.current_index - 1][0].month <= 9 or \
               self.state[self.current_index - 1][0].month >= 6:
                if self.state[self.current_index - 1][0].hour < 19 and \
                   self.state[self.current_index - 1][0].hour >= 13:
                    self.total_price += 0.127314 * total_need_grid
        elif self.charge_mode == "TOU":
            #print("Using TOU price")
            if self.state[self.current_index - 1][0].month < 11 and self.state[self.current_index - 1][0].month >= 5:
                if (self.state[self.current_index - 1][0].hour < 11 and self.state[self.current_index - 1][0].hour >= 7) or \
                (self.state[self.current_index - 1][0].hour < 19 and self.state[self.current_index - 1][0].hour >= 17):
                    self.total_price += 0.094 * total_need_grid
                if self.state[self.current_index - 1][0].hour < 17 and self.state[self.current_index - 1][0].hour >= 11:
                    self.total_price += 0.132 * total_need_grid
                if self.state[self.current_index - 1][0].hour < 7 or self.state[self.current_index - 1][0].hour >= 19:
                    self.total_price += 0.065 * total_need_grid
            if self.state[self.current_index - 1][0].month < 5 or self.state[self.current_index - 1][0].month >= 11:
                if (self.state[self.current_index - 1][0].hour < 11 and self.state[self.current_index - 1][0].hour >= 7) or \
                (self.state[self.current_index - 1][0].hour < 19 and self.state[self.current_index - 1][0].hour >= 17):
                    self.total_price += 0.132 * total_need_grid
                if self.state[self.current_index - 1][0].hour < 17 and self.state[self.current_index - 1][0].hour >= 11:
                    self.total_price += 0.095 * total_need_grid
                if self.state[self.current_index - 1][0].hour < 7 or self.state[self.current_index - 1][0].hour >= 19:
                    self.total_price += 0.065 * total_need_grid

        elif self.charge_mode == "own":
            self.total_price += self.hour_price[self.state[self.current_index - 1][0].hour]\
                                               [self.state[self.current_index - 1][0].weekday()] * \
                     total_need_grid
        # if self.current_payment == self.payment_cycle:
        #     reward = -self.total_price
        #     self.current_payment = 1
        #     self.total_price = 0
        # else:
        #     self.current_payment += 1
        reward=-self.total_price
        self.total_price = 0
        #return_state = np.copy(self.state[self.current_index][5:7])
        return_state = np.array([self.state[self.current_index][5],self.state[self.current_index][6],self.state[self.current_index][0],self.state[self.current_index][9]])
        homeid= self.datafile.split(".")[0].split("_")[3]
        # if RL:
        #     if not self.new_clf:
        #         self.new_clf = joblib.load('./saved_models/hl_rf_{}.pkl'.format(homeid))
        #     predicted_homeuse = self.new_clf.predict([[self.state[self.current_index - 1][5], self.state[self.current_index - 1][7], self.state[self.current_index - 1][4], self.state[self.current_index - 1][2], self.state[self.current_index - 1][3]]])[0]
        #     return_state[5] = predicted_homeuse
        #return_state[0] = float(return_state[0].hour)
        #return_state[9] = float(return_state[9].month)
        return total_need_grid, return_state, np.float(reward), len(self.state) == self.current_index + 1

    def reset(self):
        self.current_index = self.month_starter
        self.current_payment = 1
        self.total_price=0
        self.state[self.current_index][8] = self.battery_starter
        #return_state = np.copy(self.state[self.current_index][5:7])
        return_state = np.array([self.state[self.current_index][5],self.state[self.current_index][6],self.state[self.current_index][0],self.state[self.current_index][9]])
        # return_state[0] = float(return_state[0].hour)
        # return_state[9] = float(return_state[9].month)
        return np.hstack(return_state)

    def check_valid_action(self, action):
        current_battery = self.state[self.current_index][8]
        if current_battery + action > self.maximum_battery or current_battery + action < 0:
            return False
        else:
            return True

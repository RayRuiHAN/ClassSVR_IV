# Performed svr regression on the volatility curve, to find each order moments of RND for a given time

import numpy as np
from ClassBSM import BSM
from ClassPrep import IOPrep
from scipy.interpolate import UnivariateSpline
from sklearn.svm import SVR
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class RND(object):

    def __init__(self, input_file, input_path='.', output_path='./SVR_IV_Output', if_fig=0):
        """
        rnd.info Date
        rnd.mu, rnd.sigma, rnd.skew, rnd.kurt
        """

        self.Data = IOPrep(input_path + '/' + input_file)
        self.info = self.Data.info
        self.Date = self.Data.TradingDate
        t1 = self.Data.t1
        t2 = self.Data.t2

        valid = self.Data.IfValid
        if valid:
            x_raw = self.Data.data_iv['StrikePrice'].values
            x_raw = np.array(x_raw).reshape(-1, 1)
            y_raw = self.Data.data_iv['StdIV'].values
            y_raw = 100 * np.array(y_raw)

            x_call = self.Data.data_c['StrikePrice'].values
            x_call = np.array(x_call).reshape(-1, 1)
            y_call = self.Data.data_c['ClosePrice_C'].values
            y_call = np.array(y_call)

            x_put = self.Data.data_p['StrikePrice'].values
            x_put = np.array(x_put).reshape(-1, 1)
            y_put = self.Data.data_p['ClosePrice_P'].values
            y_put = np.array(y_put)


            # SVR model on IV
            svr_rbf = SVR(kernel='rbf', C=10)
            fit_svr = svr_rbf.fit(x_raw, y_raw)
            y_rbf = fit_svr.predict(x_raw)

            # RND Estimate
            x_l0 = np.array(range(self.Data.k.min(), self.Data.k.max() + 1))
            x_l = x_l0.reshape(-1, 1)
            y_l = 0.01 * fit_svr.predict(x_l)
            y_price1 = BSM(asset_price=self.Data.S, exercise_price=x_l0, remaining=1 / 12, sigma=y_l,
                           rf_rate=self.Data.r).Price
            y_price2 = BSM(asset_price=self.Data.S, exercise_price=x_l0, remaining=1 / 12, call_put='P', sigma=y_l,
                           rf_rate=self.Data.r).Price
            y_spl = UnivariateSpline(x_l0, y_price1, s=0, k=4)
            y_spl_2d = y_spl.derivative(n=2)
            y_f = y_spl_2d(x_l0)
            x_f = np.log(x_l0 / self.Data.S)

            if if_fig:
                plt.figure(figsize=(12, 3), dpi=200)
                plt.subplot(1, 3, 1)
                plt.scatter(x_raw, y_raw, color='darkorange', label='data')
                plt.plot(x_raw, y_rbf, color='navy', lw=2, label='RBF model')
                plt.xlabel('Strike Price')
                plt.ylabel('Target IV')
                plt.title('Support Vector Regression')
                plt.legend()

                # figure(2) Estimated Option Price
                plt.subplot(1, 3, 2)
                plt.scatter(x_call, y_call, color='navy', s=8)
                plt.scatter(x_put, y_put, color='darkorange', s=8)
                plt.plot(x_l, y_price1, color='darkorange', lw=2, label='Call')
                plt.plot(x_l, y_price2, color='navy', lw=2, label='Put')
                plt.xlabel('Strike Price')
                plt.ylabel('Price Unit')
                plt.title('Estimated Option Price')
                plt.legend()

                # figure(3) Estimated RND
                plt.subplot(1, 3, 3)
                plt.plot(x_f, y_f, color='navy', lw=2)
                plt.xlim(-0.2, 0.2)

                # y_spl2 = UnivariateSpline(x_l0, y_price2, s=0, k=4)
                # y_spl_2d2 = y_spl2.derivative(n=2)
                # plt.plot(x_f, y_spl_2d2(x_l0), color='darkorange', lw=1, label='From Call')
                plt.xlabel('Return')
                plt.ylabel('Density')
                plt.title('Estimated RND')
                # plt.legend()

                plt.tight_layout(pad=2.5)
                fig_output = output_path + '/' + input_file + '.png'
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                plt.savefig(fig_output)
                plt.clf()
                plt.close('all')

            self.mu = np.sum(x_f * y_f)
            self.sigma = np.sqrt(np.sum((x_f - self.mu) ** 2 * y_f))
            self.skew = np.sum((x_f - self.mu) ** 3 * y_f) / (self.sigma ** 3)
            self.kurt = np.sum((x_f - self.mu) ** 4 * y_f) / (self.sigma ** 4) - 3
            self.m5 = np.sum((x_f - self.mu) ** 5 * y_f) / (self.sigma ** 5)
            self.m6 = np.sum((x_f - self.mu) ** 6 * y_f) / (self.sigma ** 6)
            self.m7 = np.sum((x_f - self.mu) ** 7 * y_f) / (self.sigma ** 7)

        else:
            nan = ''
            self.mu = nan
            self.sigma = nan
            self.skew = nan
            self.kurt = nan
            self.m5 = nan
            self.m6 = nan
            self.m7 = nan


if __name__ == '__main__':
    rnd = RND("20210412.csv", input_path='./IO19-22', output_path='./001test', if_fig=1)
    print(rnd.info)
    print(rnd.mu, rnd.sigma, rnd.skew, rnd.kurt, rnd.m5, rnd.m6, rnd.m7)

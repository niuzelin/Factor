import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


class VolumeFactors:
	def __init__(self, factor_list, start_date, end_date, window, classify=False, code=None):
		"""
		在这个class增加因子时，首先要在factors_all里增加因子名称；
		同时在factors_dict增加函数调用;
		另外，如果新加入的因子需要首先对数据按日期groupby，则还要加入到{}，以激活self.df_split
		"""
		# 未来：加入数据库索引的格式，直接取出价量数据
		factors_all = {'sum_window_volume', 'until_window_volume', 'mean_window_volume', 'vol_window_pct'}
		if not set(factor_list).issubset(factors_all):
			print('输入的因子名称有误，请选取以下因子输入')
			print(factors_all)
		else:
			self.window = window
			if self.window not in ['30m', '1h', '1d', '60m', '240m']:
				print("Error: window必须为 '30m', '1h', '1d', '60m' 或 '240m' ")
			else:
				if self.window == '1h':
					self.window = '60m'
				elif self.window == '1d':
					self.window = '240m'
				self.code = code
				self.classify = classify
				# 未来这里要加入从数据库读取股票数据的代码。建议写成子函数的形式。
				# 读取时按照时间为索引；每次全部读入再取，效率会低。
				df = pd.read_csv(r'./data/000300_1min.csv', usecols=['datetime', 'vol', 'close', 'open'])
				df['date'] = df.datetime.apply(lambda x: datetime.date(int(x[:4]), int(x[5:7]), int(x[8:10])))
				df['time'] = df.datetime.apply(lambda x: datetime.time(int(x[-8:-6]), int(x[-5:-3]), int(x[-2:])))
				df = select_data(df, 'date', start_date, end_date)
				df['return_classifier'] = df.close.pct_change()
				df['return_classifier'].iloc[0] = df.close.iloc[0] / df.open.iloc[0] - 1
				self.df = df[['date', 'time', 'vol', 'return_classifier']].copy()
				del df

				if set(factor_list).union({'sum_window_volume', 'until_window_volume',
				                           'mean_window_volume', 'vol_window_pct'}) is not None:
					self.df_split = self.split_data(self.df, self.window)
					self.columns_no_date = self.df_split.columns.tolist()
					self.columns_no_date.remove('date')

	def run(self, factor_list):
		factors_fun_dict = {
			'sum_window_volume': 'self.sum_window_volume()',
			'until_window_volume': 'self.until_window_volume()',
			'mean_window_volume': 'self.mean_window_volume(self.window)',
			'vol_window_pct': 'self.vol_window_pct()'
		}
		results = []
		for iFactor in factor_list:
			results.append(eval(factors_fun_dict[iFactor]))
		return results

	def sum_window_volume(self):
		return self.df_split

	def until_window_volume(self):
		df_data = self.df_split.copy()
		for iCol in self.columns_no_date:
			index = self.columns_no_date.index(iCol)
			df_data.loc[:, iCol] = df_data.loc[:, self.columns_no_date[max(0, index-1)]: iCol].sum(axis=1)
		df_data.rename(columns={iColumn: 'until_window_volume_' + iColumn for iColumn in self.columns_no_date}, inplace=True)
		return df_data

	def mean_window_volume(self, window):
		"""分钟成交量"""
		df_data = self.df_split.copy()
		df_data.loc[:, self.columns_no_date] = df_data.loc[:, self.columns_no_date] / int(window[:-1])
		df_data.rename(columns={iColumn: 'mean_' + iColumn for iColumn in self.columns_no_date}, inplace=True)
		return df_data

	def vol_window_pct(self):
		"""量比"""
		df_data = self.df_split.copy()
		df_data[self.columns_no_date] = df_data[self.columns_no_date] / df_data[self.columns_no_date].shift()
		df_data.rename(columns={iColumn: 'vol_window_pct_' + iColumn for iColumn in self.columns_no_date}, inplace=True)
		return df_data.dropna(axis=0)

	def split_data(self, df, window):
		if window == "30m":
			df_data = df.groupby('date').apply(self.split_by_30min)
			if self.classify:
				header = [
					'vol30_1000_buy', 'vol30_1030_buy', 'vol30_1100_buy', 'vol30_1130_buy',
					'vol30_1330_buy', 'vol30_1400_buy', 'vol30_1430_buy', 'vol30_1500_buy',
					'vol30_1000_sell', 'vol30_1030_sell', 'vol30_1100_sell', 'vol30_1130_sell',
					'vol30_1330_sell', 'vol30_1400_sell', 'vol30_1430_sell', 'vol30_1500_sell']
			else:
				header = ['vol30_1000', 'vol30_1030', 'vol30_1100', 'vol30_1130',
				          'vol30_1330', 'vol30_1400', 'vol30_1430', 'vol30_1500']
			df_data.columns = header
			df_data.reset_index(level=0, inplace=True)
			return df_data
		elif window == '60m':
			df_data = df.groupby('date').apply(self.split_by_1h)
			if self.classify:
				header = [
					'vol60_1030_buy', 'vol60_1130_buy', 'vol60_1400_buy', 'vol60_1500_buy',
					'vol60_1030_sell', 'vol60_1130_sell', 'vol60_1400_sell', 'vol60_1500_sell']
			else:
				header = ['vol60_1030', 'vol60_1130', 'vol60_1400', 'vol60_1500']
			df_data.columns = header
			df_data.reset_index(level=0, inplace=True)
			return df_data
		elif window == '240m':
			return df
		else:
			print('Error: 输入的window出错，但是错误原因不明')

	def split_by_30min(self, df):
		if self.classify:
			vol30_1000_buy = df[(df.time >= datetime.time(9, 30)) & (df.time <= datetime.time(10, 0)) & (df['return_classifier'] > 0)].vol.sum()
			vol30_1030_buy = df[(df.time > datetime.time(10, 0)) & (df.time <= datetime.time(10, 30)) & (df['return_classifier'] > 0)].vol.sum()
			vol30_1100_buy = df[(df.time > datetime.time(10, 30)) & (df.time <= datetime.time(11, 0)) & (df['return_classifier'] > 0)].vol.sum()
			vol30_1130_buy = df[(df.time > datetime.time(11, 0)) & (df.time <= datetime.time(11, 30)) & (df['return_classifier'] > 0)].vol.sum()
			vol30_1330_buy = df[(df.time > datetime.time(11, 30)) & (df.time <= datetime.time(13, 30)) & (df['return_classifier'] > 0)].vol.sum()
			vol30_1400_buy = df[(df.time > datetime.time(13, 30)) & (df.time <= datetime.time(14, 0)) & (df['return_classifier'] > 0)].vol.sum()
			vol30_1430_buy = df[(df.time > datetime.time(14, 0)) & (df.time <= datetime.time(14, 30)) & (df['return_classifier'] > 0)].vol.sum()
			vol30_1500_buy = df[(df.time > datetime.time(14, 30)) & (df.time <= datetime.time(15, 0)) & (df['return_classifier'] > 0)].vol.sum()

			vol30_1000_sell = df[(df.time >= datetime.time(9, 30)) & (df.time <= datetime.time(10, 0)) & (df['return_classifier'] < 0)].vol.sum()
			vol30_1030_sell = df[(df.time > datetime.time(10, 0)) & (df.time <= datetime.time(10, 30)) & (df['return_classifier'] < 0)].vol.sum()
			vol30_1100_sell = df[(df.time > datetime.time(10, 30)) & (df.time <= datetime.time(11, 0)) & (df['return_classifier'] < 0)].vol.sum()
			vol30_1130_sell = df[(df.time > datetime.time(11, 0)) & (df.time <= datetime.time(11, 30)) & (df['return_classifier'] < 0)].vol.sum()
			vol30_1330_sell = df[(df.time > datetime.time(11, 30)) & (df.time <= datetime.time(13, 30)) & (df['return_classifier'] < 0)].vol.sum()
			vol30_1400_sell = df[(df.time > datetime.time(13, 30)) & (df.time <= datetime.time(14, 0)) & (df['return_classifier'] < 0)].vol.sum()
			vol30_1430_sell = df[(df.time > datetime.time(14, 0)) & (df.time <= datetime.time(14, 30)) & (df['return_classifier'] < 0)].vol.sum()
			vol30_1500_sell = df[(df.time > datetime.time(14, 30)) & (df.time <= datetime.time(15, 0)) & (df['return_classifier'] < 0)].vol.sum()

			df_data = pd.DataFrame([
				vol30_1000_buy, vol30_1030_buy, vol30_1100_buy, vol30_1130_buy,
				vol30_1330_buy, vol30_1400_buy, vol30_1430_buy, vol30_1500_buy,
				vol30_1000_sell, vol30_1030_sell, vol30_1100_sell, vol30_1130_sell,
				vol30_1330_sell, vol30_1400_sell, vol30_1430_sell, vol30_1500_sell]).transpose()
		else:
			vol30_1000 = df[(df.time >= datetime.time(9, 30)) & (df.time <= datetime.time(10, 0))].vol.sum()
			vol30_1030 = df[(df.time > datetime.time(10, 0)) & (df.time <= datetime.time(10, 30))].vol.sum()
			vol30_1100 = df[(df.time > datetime.time(10, 30)) & (df.time <= datetime.time(11, 0))].vol.sum()
			vol30_1130 = df[(df.time > datetime.time(11, 0)) & (df.time <= datetime.time(11, 30))].vol.sum()
			vol30_1330 = df[(df.time > datetime.time(11, 30)) & (df.time <= datetime.time(13, 30))].vol.sum()
			vol30_1400 = df[(df.time > datetime.time(13, 30)) & (df.time <= datetime.time(14, 0))].vol.sum()
			vol30_1430 = df[(df.time > datetime.time(14, 0)) & (df.time <= datetime.time(14, 30))].vol.sum()
			vol30_1500 = df[(df.time > datetime.time(14, 30)) & (df.time <= datetime.time(15, 0))].vol.sum()
			df_data = pd.DataFrame([vol30_1000, vol30_1030, vol30_1100, vol30_1130,
				 vol30_1330, vol30_1400, vol30_1430, vol30_1500]).transpose()
		return df_data

	def split_by_1h(self, df):
		if self.classify:
			vol60_1030_buy = df[(df.time >= datetime.time(9, 30)) & (df.time <= datetime.time(10, 30)) & (df['return_classifier'] > 0)].vol.sum()
			vol60_1130_buy = df[(df.time > datetime.time(10, 30)) & (df.time <= datetime.time(11, 30)) & (df['return_classifier'] > 0)].vol.sum()
			vol60_1400_buy = df[(df.time > datetime.time(11, 30)) & (df.time <= datetime.time(14, 0)) & (df['return_classifier'] > 0)].vol.sum()
			vol60_1500_buy = df[(df.time > datetime.time(14, 0)) & (df.time <= datetime.time(15, 0)) & (df['return_classifier'] > 0)].vol.sum()

			vol60_1030_sell = df[(df.time >= datetime.time(9, 30)) & (df.time <= datetime.time(10, 30)) & (df['return_classifier'] < 0)].vol.sum()
			vol60_1130_sell = df[(df.time > datetime.time(10, 30)) & (df.time <= datetime.time(11, 30)) & (df['return_classifier'] < 0)].vol.sum()
			vol60_1400_sell = df[(df.time > datetime.time(11, 30)) & (df.time <= datetime.time(14, 0)) & (df['return_classifier'] < 0)].vol.sum()
			vol60_1500_sell = df[(df.time > datetime.time(14, 0)) & (df.time <= datetime.time(15, 0)) & (df['return_classifier'] < 0)].vol.sum()
			df_data = pd.DataFrame([
				vol60_1030_buy, vol60_1130_buy, vol60_1400_buy, vol60_1500_buy,
				vol60_1030_sell, vol60_1130_sell, vol60_1400_sell, vol60_1500_sell]).transpose()
		else:
			vol60_1030 = df[(df.time >= datetime.time(9, 30)) & (df.time <= datetime.time(10, 30))].vol.sum()
			vol60_1130 = df[(df.time > datetime.time(10, 30)) & (df.time <= datetime.time(11, 30))].vol.sum()
			vol60_1400 = df[(df.time > datetime.time(11, 30)) & (df.time <= datetime.time(14, 0))].vol.sum()
			vol60_1500 = df[(df.time > datetime.time(14, 0)) & (df.time <= datetime.time(15, 0))].vol.sum()
			df_data = pd.DataFrame([vol60_1030, vol60_1130, vol60_1400,  vol60_1500]).transpose()
		return df_data


def volume_factor(factor_list, start_date, end_date, window, classify=False, code=None):
	if isinstance(factor_list, str):
		factor_list = [factor_list]
	class_volume_factors = VolumeFactors(factor_list, start_date, end_date, window, classify, code)
	results = class_volume_factors.run(factor_list)
	if len(results) == 1:
		return results
	else:
		df = results[0].copy()
		for iIndex in np.arange(len(results)-1):
			df = pd.merge(df, results[iIndex + 1], on='date')
		return df


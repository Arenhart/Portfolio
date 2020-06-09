# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:16:06 2019

@author: Arenhart
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.svm import LinearSVR, SVR


lines = []
sample_names = []

with open('all_data.txt', mode = 'r') as file:
	for line in file:
		lines.append(line)

h = lines[0].split('\t')
h[-1] = h[-1][:-1]

data_arr = np.zeros((len(lines)-1, len(h)-1), dtype = 'float32')
y_exp = []
experimental_permeability = {'A1' : 1.091,
							 'A2' : 0.727,
							 'A3' : 0.11,
							 'A4' : 0.042,
							 'A5' : 0.018,
							 'A6' : 0.004 }

for index, line in enumerate(lines[1:]):
	data = line.split('\t')
	sample_names.append(data[0])
	for i, val in enumerate(data[1:]):
		data_arr[index][i] = val
	y_exp.append(experimental_permeability[data[0][:2]])
	
y_exp = np.array(y_exp)
y_exp = np.log(y_exp)	
X = data_arr[:,1]
y = data_arr[:,0]
log_y = np.log(y)

scores_m = []
#for i in range(20)
reg = LinearRegression().fit(X.reshape(-1, 1), log_y)
coef = reg.coef_[0]
intercept = reg.intercept_

scores = cross_val_score(reg, X.reshape(-1, 1), log_y, cv=KFold(n_splits=20, shuffle = True))
print('Lin Reg', scores.mean(), scores.std())

'''
Lin Reg 0.4647877374967077
'''

#plt.plot(X, log_y, 'o')

features = data_arr[:,1:]
value = log_y
k = 5

selected_features = SelectKBest(mutual_info_regression, k=4)
sel_features = selected_features.fit_transform(features, log_y)
#print([i for i in zip(selected_features.get_support(),h[2:])])

#X_train, X_test, y_train, y_test = train_test_split(sel_features, log_y, test_size=0.2, random_state=0)

svm = LinearSVR(max_iter=10000000)#.fit(X_train, y_train)
#svm = SVR(max_iter=100000000, kernel = 'rbf').fit(X_train, y_train)
#svm = SVR(max_iter=100000000, kernel = 'poly').fit(X_train, y_train)
for i in range(4,5):
	selected_features = SelectKBest(mutual_info_regression, k=i)
	sel_features = selected_features.fit_transform(features, log_y)
	sco = cross_val_score(svm, sel_features, log_y, cv=KFold(n_splits=20, shuffle = True))
	print(i, sco.mean(), sco.std())

#res = svm.predict(X_test)
#plt.plot(y_test, res, 'o')


'''
with cv = 10
2 0.3698684507725337
3 0.6017871163153103
4 0.6057979221576391 ***
5 0.6041074994170599
6 0.6012144471357697
7 0.6058371622933331
8 0.5943724741148799
9 0.5980129837115857
10 0.5843999968549871
11 0.5889230138077081
12 0.5900719082044603
13 0.589610941967806
14 0.5916299235446159
15 0.6011922423312434
16 0.5842347250176123
17 0.5864758706073439
17 0.5854220287085102
18 0.5879873894425538
19 0.5881194750746916
20 0.6061054185323596 ***
21 0.6060458769166607
22 0.5925791859592138
23 0.5799534790513524
24 0.5799790050999241

with cv = 20
2 0.2985856918249675
3 0.5659355589426042
4 0.5677483995398513
5 0.5646439799166043
6 0.5615862548661574
7 0.5662989233375572
8 0.5526892729928137
9 0.5627290492526346
10 0.5489578822878555
11 0.5521048489477933
12 0.5458680180527711
13 0.5453972944511768
14 0.5460943493535292
15 0.5545582648455385
16 0.5264210386122402
17 0.5308610083261415
18 0.534174170727687
19 0.5244491654453273
20 0.5339468157276087
21 0.5350144197886426
22 0.5185546835928025
'''

r2_lsvr = [
0.2985856918249675,
0.5659355589426042,
0.5677483995398513,
0.5646439799166043,
0.5615862548661574,
0.5662989233375572,
0.5526892729928137,
0.5627290492526346,
0.5489578822878555,
0.5521048489477933,
0.5458680180527711,
0.5453972944511768,
0.5460943493535292,
0.5545582648455385,
0.5264210386122402,
0.5308610083261415,
0.534174170727687,
0.5244491654453273,
0.5339468157276087,
0.5350144197886426,
0.5185546835928025]

fig, ax = plt.subplots()

ax.plot(range(2,23), r2_lsvr)
ax.set_ylabel('Escore [R^2]')
ax.set_xlabel('Número de atributos')


plt.show()


lr_pred = cross_val_predict(reg, X.reshape(-1, 1), log_y, cv=KFold(n_splits=5, shuffle = True))

selected_features = SelectKBest(mutual_info_regression, k=4)
sel_features = selected_features.fit_transform(features, log_y)
svr4_pred = cross_val_predict(svm, sel_features, log_y, cv=KFold(n_splits=5, shuffle = True))

'''
selected_features = SelectKBest(mutual_info_regression, k=20)
sel_features = selected_features.fit_transform(features, log_y)
svr20_pred = cross_val_predict(svm, sel_features, log_y, cv=20)
'''

'''
fig, ax = plt.subplots()
ax.scatter(lr_pred, log_y, c = 'red')
ax.scatter(svr4_pred, log_y, c = 'green')
#ax.scatter(svr20_pred, log_y, c = 'blue')
ax.plot([log_y.min(), log_y.max()], [log_y.min(), log_y.max()], 'k--', lw=4)

plt.show()
'''


fig, ax = plt.subplots()
ax.set_xticklabels(['Regressão Linear', 'SVR 4', 'SVR 20'],rotation=45, fontsize=8)
data = [(lr_pred-log_y)**2, (svr4_pred-log_y)**2]#, (svr20_pred-log_y)**2]
ax.boxplot(data, 0, '')
ax.set_ylabel('Variância da Permeabilidade [uD^2]')

plt.show()


'''
reg = LinearRegression().fit(X.reshape(-1, 1), y_exp)
coef = reg.coef_[0]
intercept = reg.intercept_
#plt.plot(y_exp,  reg.predict(X.reshape(-1, 1)), 'o')

scores = cross_val_score(reg, X.reshape(-1, 1), y_exp, cv=KFold(n_splits=5, shuffle = True))
print('Lin Reg', scores.mean())


Lin Reg 0.20801886350108073


features = data_arr[:,1:]
value = y_exp
svm = LinearSVR(max_iter=100000000)

for i in range(1,25):
	selected_features = SelectKBest(mutual_info_regression, k=i)
	sel_features = selected_features.fit_transform(features, value)
	scores = cross_val_score(svm, sel_features, value, cv=KFold(n_splits=5, shuffle = True))
	print(i, scores.mean())
'''
'''
cv = 5
1 -0.08457128185020939
2 0.38529224024640174
3 0.6216058759349364 
4 0.624678808163051
5 0.6182785269834161
6 0.6215207634221788
7 0.627770962889865 ***
8 0.6073199219751235
9 0.6084855191219912
10 0.6060594783258111
11 0.6097605747955436
12 0.607860124674818
13 0.6068844488706734
14 0.6097066641778586
15 0.6174049039085722
16 0.6235921396172961
17 0.630114223981008
18 0.631886061246908
19 0.6323511652530045
20 0.6318458799480837
21 0.6326495012801211 ***
22 0.6203860720371721
23 0.626099334217429
24 0.6264522270435091
'''
'''
r2_lsvr_exp = [
0.38529224024640174,
0.6216058759349364 ,
0.624678808163051,
0.6182785269834161,
0.6215207634221788,
0.627770962889865,
0.6073199219751235,
0.6084855191219912,
0.6060594783258111,
0.6097605747955436,
0.607860124674818,
0.6068844488706734,
0.6097066641778586,
0.6174049039085722,
0.6235921396172961,
0.630114223981008,
0.631886061246908,
0.6323511652530045,
0.6318458799480837,
0.6326495012801211,
0.6203860720371721,
0.626099334217429,
0.6264522270435091]

fig, ax = plt.subplots()

ax.plot(range(2,25), r2_lsvr_exp)
ax.set_ylabel('Escore [R^2]')
ax.set_xlabel('Número de atributos')


plt.show()

selected_features = SelectKBest(mutual_info_regression, k=7)
sel_features = selected_features.fit_transform(features, log_y)
#print([i for i in zip(selected_features.get_support(),h[2:])])

lr_pred = cross_val_predict(reg, X.reshape(-1, 1), log_y, cv=5)

#selected_features = SelectKBest(mutual_info_regression, k=7)
#sel_features = selected_features.fit_transform(features, log_y)
#svr4_pred = cross_val_predict(svm, sel_features, log_y, cv=5)
'''
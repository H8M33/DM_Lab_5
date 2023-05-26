import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

url = 'https://drive.google.com/file/d/1UGJhTd01_AmzcTnZxiySE5bOGaHkcm8h/view'
file_id = url.split('/')[-2]
dwn_url = 'https://drive.google.com/uc?id=' + file_id
data = pd.read_csv(dwn_url)

# рассмотрим зависимость среднего числа очков точности бросков (`pts`)
# от меры эффективности броска игрока (`ts_pct`) и количества игр игрока
# за сезон (`gp`) с помощью линейной, полиномиальной регрессии и SVM.

X = data.iloc[:, [12, 20]].values  # gp, ts_pct
y = data.iloc[:, 13].values  # pts

# linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
lin_reg_score = r2_score(y_test, y_pred)

# polynomial regression
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=0)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_train, y_train)
y_pred_poly = lin_reg_2.predict(X_test)
poly_reg_score = r2_score(y_test, y_pred_poly)

# SVM
svr_rbf_reg = SVR(kernel='rbf')
svr_lin_reg = SVR(kernel='linear')
svr_poly_reg = SVR(kernel='poly')
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)  # standartization
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
svr_rbf_reg.fit(X_train, y_train)
svr_lin_reg.fit(X_train, y_train)
svr_poly_reg.fit(X_train, y_train)
y_pred_svr_rbf = svr_rbf_reg.predict(X_test)
y_pred_svr_lin = svr_lin_reg.predict(X_test)
y_pred_svr_poly = svr_poly_reg.predict(X_test)
svr_rbf_score = r2_score(y_test, y_pred_svr_rbf)
svr_lin_score = r2_score(y_test, y_pred_svr_lin)
svr_poly_score = r2_score(y_test, y_pred_svr_poly)

# Построим графики зависимости среднего числа очков точности бросков
# от меры эффективности броска игрока (`ts_pct`) и количества игр игрока
# за сезон (`gp`) с помощью линейной, полиномиальной регрессии и SVM.

# linear regression plot
plt.scatter(X_test[:, 1], y_test, color='red')
plt.plot(X_test[:, 1], y_pred, color='blue')
plt.title('Linear Regression')
plt.xlabel('ts_pct')
plt.ylabel('pts')
plt.show()

# polynomial regression plot
plt.scatter(X_test[:, 1], y_test, color='red')
plt.plot(X_test[:, 1], y_pred_poly, color='blue')
plt.title('Polynomial Regression')
plt.xlabel('ts_pct')
plt.ylabel('pts')
plt.show()

# SVR kernel plot
plt.scatter(X_test[:, 1], y_test, color='red')
plt.plot(X_test[:, 1], y_pred_svr_rbf, color='green', label='RBF')
plt.plot(X_test[:, 1], y_pred_svr_lin, color='blue', label='Linear')
plt.plot(X_test[:, 1], y_pred_svr_poly, color='red', label='Polynomial')
plt.title('SVR Regression')
plt.xlabel('ts_pct')
plt.ylabel('pts')
plt.legend()
plt.show()

# Оценка `score` моделей:

print(f'Linear Regression score: {lin_reg_score:.4f}')
print(f'Polynomial Regression score: {poly_reg_score:.4f}')
print(f'SVR RBF kernel score: {svr_rbf_score:.4f}')
print(f'SVR Linear kernel score: {svr_lin_score:.4f}')
print(f'SVR Polynomial kernel score: {svr_poly_score:.4f}')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import qmc, truncnorm, norm
import time

def f(x):
    return np.min([2 * x**2 + 2 * a**2 + 1 + 2 * x * a for a in [-1, 0, 1]])

def f_dt(x, model, ran_arr, act_arr = [-1, 0, 1]):
    return np.min([x**2 + a**2 + np.mean([model.predict([[x + a + xi]])[0] for xi in ran_arr]) for a in act_arr])

def c(x, a):
    return x**2 + a**2

def sobol_normal(n, dim=1, seed = 47):
    sampler = qmc.Sobol(d=dim, scramble=True, seed = seed)
    sample = sampler.random(n)
    normal_sample = norm.ppf(sample)
   
    return normal_sample.ravel()

T = 1
N = 32
x0 = 1
act_arr = [-1, 0, 1]
ran_arr = sobol_normal(N)
V = {}

means = [[x0]]

start = time.time()

for t in range(1, T+1):
    tmp = list(range(means[t-1][0] - 1, means[t-1][-1] + 2))
    means.append(tmp)

park = [DecisionTreeRegressor() for t in range(T + 1)]
ran_arr = sobol_normal(N)

for t in range(T, -1, -1):
    X = np.array([ran_arr + mu for mu in means[t]]).reshape(-1, 1)
    y = []
    if t == T:
        y = np.array([x[0]**2 for x in X])
    else:
        y = np.array([f_dt(x[0], park[t+1], ran_arr) for x in X])

    park[t].fit(X, y)

end = time.time()

print(f'Time for Training the Predictors: {end - start} s.')

X = np.arange(-3, 3, 0.05)
y_pred = park[0].predict(X.reshape(-1, 1))
y_true = np.array([f(x) for x in X])

plt.plot(X, y_pred, label = 'prediction')
plt.plot(X, y_true, label = 'true function')
plt.legend()
plt.xlabel(f'$x_0$')
plt.ylabel(f'$V(0, x_0)$')
plt.title(f'Value function at time $t=0$ for $T=1$')
plt.savefig('Value Function Graph.png')

for x0 in [0, 1, 2]:
    pred = park[0].predict([[x0]])[0]
    true = f(x0)
    abs_err = np.abs(pred - true)
    rel_err = np.abs(abs_err / true)
    
    print(f'x_0={x0}')
    print(f'Prediction: {pred}')
    print(f'True answer: {true}')
    print(f'Absolute Error: {abs_err}')
    print(f'Relative Errpr: {rel_err}')
    print()

from math import sin, log, exp
from numpy import arange
import numpy as np 
import matplotlib.pyplot as plt
import random
from itertools import product


def hill_climb(x_0, step_size):

	current = x_0
	E_curr = Y(x_0)
	iterations = 0
	E_max = E_curr
	i_max = current

	while(True):

		# get neighbours, i.e. one step size to the left and right since Y is a 2D function
		left = current - step_size
		right = current + step_size
		if Y(left) > Y(right) and left > 0.0:
			E_max = Y(left)
			i_max = left
		elif Y(right) > Y(left) and right < 10.0:
			E_max = Y(right)
			i_max = right

		# if max valued neighbour is less than current, we have found a local minimum.
		if E_max <= E_curr:
			return current, E_curr, iterations
		else:
			iterations += 1
			current = i_max
			E_curr = E_max


def SimulatedAnnealing(x_0, step_size, alpha, temperature):

	X_max = x_0
	E_max = Y(x_0)
	X = x_0
	E = Y(x_0)
	iterations = 0
	T = temperature

	# very ugly and hacky solution to prevent overflow errors 
	while(T > 0.000000000000000000000000000000000000000000000000000000000000000001):

		iterations += 1

		# gradually decrease the temperature by a constant factor alpha
		if T * alpha > 0.0 : #
			T = T * alpha
		else: 
			T = 0

		# get neighbours, i.e. one step size to the left and right since Y is a 2D function
		if X - step_size > 0.0: 
			left = X - step_size
		else: 
			left = 0.0
		if X + step_size < 10.0: 
			right = X + step_size
		else: 
			right = 10.0

		# randomly choose the neighbour of X
		X_i = random.choice([left,right])
		if Y(X_i) > E_max:
			X_max = X_i
			E_max = Y(X_i)
		if Y(X_i) > E:
			X = X_i
			E = Y(X_i)
		else:
			# with some probability p, still accept the 
			p = exp(-(E - Y(X_i)) / T)
			if random.random() <= p:
				X = X_i
				E = Y(X_i)
	return X_max, E_max, iterations

def plotSimulatedAnnealing():

	optima_se = []
	alphas = [0.1, 0.5, 0.99]
	step_sizes = [0.1, 0.9]
	temperatures = [100, 10000]
	starting_points = list(range(11))
	Y = lambda x: np.sin(x**2 / 2) / np.log2(x + 4)
	params = list(product(*[starting_points, step_sizes, alphas, temperatures]))
	params = list(product(*[starting_points, [0.9], [0.99], [100]]))

	for (p,s,a,t) in params:
		x, y, steps = SimulatedAnnealing(p, s, a, t)
		print 'starting point:',p, 'alpha', a, 'step size:', s, 'temp:', t, 'X:', x, 'Y:', y, 'steps until convergence:', steps
		optima_se.append((x,y))


	xvals = arange(0.0, 10.0, 0.01)
	yvals = Y(xvals)
	plt.plot(xvals, yvals)
	for (x,y) in optima_se:
		plt.plot(x,y, 'ro')
	plt.show()

def plotHillClimbing():

	optima_hc = []
	starting_points = list(range(11))
	Y = lambda x: np.sin(x**2 / 2) / np.log2(x + 4)
	step_sizes = [x * 0.01 for x in range(1, 10)]
	params = list(product(*[starting_points, step_sizes]))

	for (p,s) in params:
		x, y, steps = hill_climb(p, s)
		print 'starting point:',p, 'step size:', s, 'X:', x, 'Y:', y, 'steps until convergence:', steps
		optima_hc.append((x,y))

	xvals = arange(0.0, 10.0, 0.01)
	yvals = Y(xvals)
	plt.plot(xvals, yvals)
	for (x,y) in optima_hc:
		plt.plot(x,y, 'ro')
	plt.show()


if __name__ == "__main__":

	plotHillClimbing()
	plotSimulatedAnnealing()
	


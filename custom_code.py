
# ========== Packages ==========
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Functions ==========
def plot_continous_onecol(df, feature, bins=30, figsize=(14, 5)):
	"""Plot histogram, density plot, box plot and swarm plot for feature.
	Parameters
	----------
	df: A pandas DataFrame to use
	feature: A string specifying the name of the feature column
	bins: (optional) An integer for number of bins in histogram
	figsize: (optional) A tuple specifying the shape of the plot
	Returns
	-------
	A plot containing 4 subplots. Top left shows histogram. Top right shows density plot. Bottom left shows box plot. Bottom right shows swarm plot.
	"""
	fig, ax = plt.subplots(1, 2, figsize=(7,5))

	sns.histplot(data=df, x=feature, bins=bins, ax=ax[0])
	ax[0].set_title(f'Histogram of {feature.title()}')

	sns.boxplot(data=df, y=feature, ax=ax[1])
	ax[1].set_title(f'Box plot of {feature.title()}')

	plt.tight_layout() # To ensure subplots don't overlay

def forward_feature_selection(model):
	"""Conduct forward feature selection to assist in picking a good model. It'll fit and score a model, progressively adding features until further model performance isn't found.

	Note that this assumes the use of X & X_train notation.

	Can be adjusted for other models, by adjusting the scoring metrics.
	
	Paramaters
	----------
	model: A class specifying which estimator is being used
	Returns
	-------
	Printed output of features to be included.
	"""

	# List to store features
	included = []

	# Dictionary to track of feature and parameters
	best = {'feature': '', 'r2': 0, 'a_r2': 0}

	# Instantiate model
	model = model

	# Sample size for adj r2
	n = X_train.shape[0]

	while True:
		changed = False

		# Features to be evaluated
		excluded = list(set(X.columns) - set(included))

		# Evaluate features
		for new_column in excluded:

			# Fit model with training data
			fit = model.fit(X_train[included + [new_column]], y_train)

			# Calculate the score
			r2 = fit.score(X_train[included + [new_column]], y_train)

			# Number of predictors
			k = len(included + [new_column])

			# Adjusted R^2
			adjusted_r2 = 1 - ( ( (1 - r2) * (n - 1) ) / (n - k - 1) )

			# Model improvement
			if adjusted_r2 > best['a_r2']:

				# Record new parameters
				best = {'feature': new_column, 'r2': r2, 'a_r2': adjusted_r2}

				# Flag new better model
				changed = True

		# END

		# If found a better model after testing all remaining features
		if changed:

			# Update control details
			included.append(best['feature'])
			excluded = list(set(excluded) - set(best['feature']))
			print('Added feature %-4s with R^2 = %.3f and adjusted R^2 = %.3f' %(best['feature'], best['r2'], best['a_r2']))

		else:
			# Terminate if no better model
			print('*'*50)
			break

	print('')
	print('Resulting features:')
	print(', '.join(included))
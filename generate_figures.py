
from __future__ import division

import os
from itertools import izip

import numpy as np

from scipy.io import loadmat

import matplotlib
"""
The right matplotlib backend will need to be selected for your system.
"""
matplotlib.use("pdf")


import matplotlib.pyplot as plt

from dba import dba, dtw
# from hier import cluster_timeseries

ROOT = 'data'

SEQUENCES = os.path.join(ROOT, 'sequences.mat')
LPS = os.path.join(ROOT, 'lps_conditions.mat')
TNF = os.path.join(ROOT, 'tnf_conditions.mat')

OUTPUT = 'out'

DATA_CUTOFF = 100 # last time point

def load(remove_empty_frames = True):
	sequences = loadmat(SEQUENCES)["Mat_JohnDTW"][:, :DATA_CUTOFF]
	lps = loadmat(LPS)["MatLPS"].flatten()
	tnf = loadmat(TNF)["MatTNF"].flatten()

	if remove_empty_frames:
		sequences = [
			sequence[~np.isnan(sequence)] for sequence in sequences
			]

		valid = [
			np.all(np.isfinite(sequence)) and (sequence.size > 0)
			for sequence in sequences
			]

		sequences = [s for (is_valid, s) in izip(valid, sequences) if is_valid]
		lps = [c for (is_valid, c) in izip(valid, lps) if is_valid]
		tnf = [c for (is_valid, c) in izip(valid, tnf) if is_valid]

	n_sequences = len(sequences)

	conditions = np.empty(n_sequences, dtype = [
		("lps", np.float64),
		("tnf", np.float64)
		])

	conditions["lps"] = lps
	conditions["tnf"] = tnf

	return sequences, conditions

FIG_SIZE = (11, 8.5)

def plot_condition_averages(sequences, conditions, warp = True):
	unique_conditions = np.unique(conditions)

	if warp:
		centroid_function = lambda sequences: dba(sequences)[0]

	else:
		centroid_function = lambda sequences: np.nanmean(sequences, 0)

	centroids = [
		centroid_function([
			sequences[i]
			for i in np.where(condition == conditions)[0]
			])
		for condition in unique_conditions
		]

	plt.figure(0, figsize = FIG_SIZE)

	plt.figure(1, figsize = FIG_SIZE)

	lps_conc = np.unique(conditions["lps"])
	tnf_conc = np.unique(conditions["tnf"])

	n_rows = lps_conc.size
	n_cols = tnf_conc.size

	axes = [
		0,
		min([centroid.size for centroid in centroids]),
		0,
		2
		]

	for condition, centroid in izip(unique_conditions, centroids):

		subplot_index = np.where(condition["tnf"] == tnf_conc)[0][0] + np.where(condition["lps"] == lps_conc)[0][0]*n_cols + 1

		plt.figure(0)
		plt.subplot(n_rows, n_cols, subplot_index)

		color = "k"

		if condition["tnf"] == 0 and condition["lps"] != 0:
			color = "b"

		elif condition["lps"] == 0 and condition["tnf"] != 0:
			color = "r"

		plt.plot(centroid, color = color)

		if condition["tnf"] != 0 and condition["lps"] != 0:
			plt.plot(centroids[np.where((unique_conditions["tnf"] == 0) & (unique_conditions["lps"] == condition["lps"]))[0][0]], "b", alpha = 0.5)

			plt.plot(centroids[np.where((unique_conditions["lps"] == 0) & (unique_conditions["tnf"] == condition["tnf"]))[0][0]], "r", alpha = 0.5)

		plt.figure(1)
		plt.subplot(n_rows, n_cols, subplot_index)

		indexes = np.where(condition == conditions)[0]

		alpha = max(min(20./indexes.size, 1), 0.025)

		for index in indexes:
			sequence = sequences[index]

			if warp:
				(error, alignment1, alignment2) = dtw(centroid, sequence)

				plotted = (
					np.bincount(alignment1, sequence[alignment2]) /
					np.bincount(alignment1, None)
					)

			else:
				plotted = sequence

			plt.plot(sequence, color = "k", alpha = alpha)

		plt.plot(centroid, color = "r")

		for fig_index in xrange(2):
			plt.figure(fig_index)

			plt.axis(axes)

			plt.title(
				"LPS = {}, TNF = {}".format(condition["lps"], condition["tnf"]),
				size = 8
				)

			ax = plt.gca()

			# ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)

			plt.xticks([0, 32, 66, 99], [0, 100, 200, 300])
			ax.tick_params(axis='x', which='both', direction='out')

	plt.tight_layout()

	postfix = '' if warp else '_no_warping'

	plt.figure(0)
	plt.savefig(os.path.join(OUTPUT, "averages{}.pdf".format(postfix)))

	plt.figure(1)
	plt.savefig(os.path.join(OUTPUT, "traces{}.pdf".format(postfix)))

	plt.close("all")

def main():
	for do_dtw in (False, True):
		sequences, conditions = load(do_dtw)

		plot_condition_averages(sequences, conditions, do_dtw)

if __name__ == "__main__":
	main()

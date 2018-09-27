
from __future__ import division

from warnings import warn
from itertools import izip
from functools import partial

import numpy as np

from dtw import dtw as _dtw

_RELATIVE_TOLERANCE = 1e-2
_MAX_ITERS = 15
_WARP_PENALTY = 2

dtw = partial(_dtw, warp_penalty = _WARP_PENALTY)

def dba(sequences, size = None, tol = _RELATIVE_TOLERANCE, max_iters = _MAX_ITERS):
	"""
	Performs DTW barycenter averaging on an iterable of sequences.

	Returns: center, errors
	"""
	if size is None:
		size = max(sequence.shape[0] for sequence in sequences)

	ndim = sequences[0].ndim

	center = np.zeros(size)

	last_error = np.inf

	previous_centers = []
	previous_errors = []

	for i in xrange(max_iters):
		new_center, errors = _update_center(center, sequences)

		error = np.sum(np.square(errors))

		delta_error = abs(1 - error / last_error)
		delta_center = np.linalg.norm(center - new_center, 2) / np.linalg.norm(center, 2)

		if delta_error < tol or np.allclose(center, new_center):# or delta_center < tol:
			break

		if any(np.allclose(new_center, previous) for previous in previous_centers):
			best = np.argmin([np.sum(np.square(ers)) for ers in previous_errors])

			center = previous_centers[best]
			errors = previous_errors[best]

			break

		else:
			previous_centers.append(center)
			previous_errors.append(errors)

			center = new_center
			last_error = error

	else:
		warn(
			"failed to converge (dE/E = {}, d|v|/|v| = {})".format(delta_error, delta_center)
			)

	return center, errors

# new method that combines k-means with the DBA algorithm
def k_dba(sequences, k, tol = _RELATIVE_TOLERANCE, max_iters = 30):
	n_sequences = len(sequences)

	size = max(len(sequence) for sequence in sequences)

	weights = np.random.random((k, len(sequences)))
	weights /= weights.sum(0)

	errors = np.empty_like(weights)

	centers = [
		_update_center(np.zeros(size), sequences, center_weights)[0]
		for center_weights in weights
		]

	for iteration in xrange(max_iters):
		for i, (center, center_weights) in enumerate(izip(centers, weights)):
			new_center, new_errors = _update_center(center, sequences, center_weights)

			errors[i, :] = new_errors

			centers[i] = new_center

		weights.fill(0)
		weights[np.argmin(errors, 0), np.arange(n_sequences)] = 1

		weights[np.isnan(weights)] = 0
		weights[np.isinf(weights)] = 1

	return centers, errors, weights

def _update_center(center, sequences, weights = None):
	accumulated = np.zeros_like(center)
	n_sequences = len(sequences)
	errors = np.empty(n_sequences)

	if weights is None:
		weights = np.ones_like(errors)

	minsize = center.shape[0]

	n_observations = np.zeros(center.shape[0], np.float64)

	for i, (sequence, weight) in enumerate(izip(sequences, weights)):
		dist, alignment1, alignment2 = dtw(
			center,
			sequence,
			)

		aligned = (
			np.bincount(alignment1, sequence[alignment2], minsize) /
			np.bincount(alignment1, None, minsize)
			)

		aligned[np.isnan(aligned)] = 0

		errors[i] = dist

		accumulated += weight*aligned

		n_observations += weight*(np.bincount(alignment1, None, minsize) > 0)

	new_center = (accumulated.T / n_observations).T

	new_center[np.isnan(new_center)] = 0

	return new_center, errors

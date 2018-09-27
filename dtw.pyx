
from __future__ import division

import numpy as np

cimport numpy as np

np.import_array()

cimport cython

# Numbers for path navigation
cdef np.int8_t _NONE = -2
cdef np.int8_t _START = -1
cdef np.int8_t _DIAGONAL = 0
cdef np.int8_t _SKIP2 = 1
cdef np.int8_t _SKIP1 = 2

# TODO: windowing
# TODO: other options for norms
# TODO: profile
# TODO: tests
# TODO: options for aligning left or right ends
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef dtw(
		np.ndarray[np.float64_t, ndim = 1] seq1,
		np.ndarray[np.float64_t, ndim = 1] seq2,
		int alignment = True,
		int align_overhangs = False,
		float warp_penalty = 1.0
		):
	"""
	Finds the distance between two sequences using dynamic time warping, and
	optionally, the optimal pair of alignments.

	Parameters
	----------
	seq1 : ndarray (float)
		First sequence.
	seq2 : ndarray (float)
		Second sequence.
	alignment : bool, optional
		If true, returns the alignments.  Default is True.
	align_overhangs : bool, optional
		If true, the full extent of the longer sequence is aligned to the
		shorter sequence.  Default is False.  As a rule of thumb, it is good to
		align overhangs if the beginning and end of a sequence are of critical
		importance.  For example, if the sequences represent full cell cycles.
	warp_penalty : float, optional
		A value >= 1.0, up to inf.  Default is 1.0.  This parameter introduces
		a penalty for distorting the sequence.

	Returns
	-------
	dist : float
		The "distance" between the two sequences, after alignment.  This value
		includes any warping penalty.
	alignment1 : ndarray (int)
		A vector of indexes to the first sequence, after alignment.  Only
		returned if alignment = True.
	alignment2 : ndarray (int)
		A vector of indexes to the second sequence, after alignment.  Only
		returned if alignment = True.

	"""

	assert warp_penalty >= 1.0, "warp_penalty should be at least 1.0"

	# Inverting the warp penalty allows for feasible evaluation of the limit
	# where the penalty for warping is infinitely large
	cdef float inv_penalty = 1./warp_penalty

	cdef int size1 = seq1.size
	cdef int size2 = seq2.size

	# costs and paths are the matrices of the steps taken in the dynamic
	# programming algorithm
	cdef np.ndarray[np.float64_t, ndim = 2] costs = np.empty(
		(size1, size2), np.float64
		)

	cdef np.ndarray[np.int8_t, ndim = 2] paths = np.empty(
		(size1, size2), np.int8
		)

	# Penalty of the first step
	costs[0, 0] = inv_penalty*diff(seq1[0], seq2[0])
	paths[0, 0] = _START

	cdef int i, j, action
	cdef float c_diag, c_skip1, c_skip2, action_cost

	cdef int size_diff = abs(size1 - size2)
	cdef int seq1_larger = (size1 > size2)

	# Penalties for warping along the first elements
	for j in range(1, size2):
		if not align_overhangs and not seq1_larger and j <= size_diff:
			costs[0, j] = inv_penalty*diff(seq1[0], seq2[j])
			paths[0, j] = _START

		else:
			costs[0, j] = costs[0, j-1] + diff(seq1[0], seq2[j])
			paths[0, j] = _SKIP2

	for i in range(1, size1):
		if not align_overhangs and seq1_larger and i <= size_diff:
			costs[i, 0] = inv_penalty*diff(seq1[i], seq2[0])
			paths[i, 0] = _START

		else:
			costs[i, 0] = costs[i-1, 0] + diff(seq1[i], seq2[0])
			paths[i, 0] = _SKIP1

	# All other penalties
	for i in range(1, size1):
		for j in range(1, size2):
			c_diag = costs[i-1, j-1]
			c_skip2 = costs[i, j-1]
			c_skip1 = costs[i-1, j]

			if c_diag <= c_skip2:
				if c_diag <= c_skip1:
					action = _DIAGONAL
					action_cost = c_diag

				else:
					action = _SKIP1
					action_cost = c_skip1

			else:
				if c_skip1 <= c_skip2:
					action = _SKIP1
					action_cost = c_skip1

				else:
					action = _SKIP2
					action_cost = c_skip2

			paths[i, j] = action

			if action == _DIAGONAL:
				costs[i, j] = action_cost + inv_penalty*diff(seq1[i], seq2[j])

			else:
				costs[i, j] = action_cost + diff(seq1[i], seq2[j])

	cdef float smallest_cost = np.inf
	cdef int index, smallest_index

	# Walk backwards along the path matrix to generate the alignment vectors
	if align_overhangs:
		i, j = size1-1, size2-1

	else:
		if seq1_larger:
			for index in range(size1 - size_diff - 1, size1):
				if costs[index, size2-1] < smallest_cost:
					smallest_cost = costs[index, size2-1]
					smallest_index = index
			i = smallest_index

			j = size2-1

		else:
			i = size1-1

			for index in range(size2 - size_diff - 1, size2):
				if costs[size1-1, index] < smallest_cost:
					smallest_cost = costs[size1-1, index]
					smallest_index = index
			j = smallest_index

	# Compute the distance bewteen sequences (root-sum-squared-difference)
	cdef float dist = np.sqrt(warp_penalty*costs[i, j])

	# If the alignment isn't important, just return the distance
	if not alignment:
		return dist

	cdef int max_align_size = size1 + size2

	cdef np.ndarray[np.int64_t, ndim = 1] alignment1 = np.empty(max_align_size, np.int64)
	cdef np.ndarray[np.int64_t, ndim = 1] alignment2 = np.empty(max_align_size, np.int64)

	action = _NONE

	index = max_align_size-1

	while action != _START:
		alignment1[index] = i
		alignment2[index] = j

		action = paths[i, j]

		if action == _DIAGONAL:
			i -= 1
			j -= 1

		elif action == _SKIP2:
			j -= 1

		elif action == _SKIP1:
			i -= 1

		index -= 1

	return dist, alignment1[index+1:], alignment2[index+1:]


cpdef inline float diff(float value1, float value2):
	return (value1 - value2)*(value1 - value2)
	# return abs_float(value1 - value2)


cpdef inline float abs_float(float value):
	if value > 0:
		return value

	else:
		return -value


cpdef inline int abs_int(int value):
	if value > 0:
		return value

	else:
		return -value

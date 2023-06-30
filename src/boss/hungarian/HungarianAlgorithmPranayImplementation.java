package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;

public class HungarianAlgorithmPranayImplementation extends HungarianAlgorithmPranay {
	double cost;

	/**
	 * XXX - Copy: why? Really required?X
	 * 
	 * @param matrix_to_copy
	 * @param nRows
	 * @param nCols
	 * @return
	 */
	public static final double[][] copy_and_transpose(final double[][] matrix_to_copy, final int nRows,
			final int nCols) {
		double[][] distMatrixCopy = new double[nCols][nRows];//
		// Fill in the distMatrixIn. Mind the index is "i + nRows * j".
		// Here the cost matrix of size MxN is defined as a double precision array of
		// N*M elements.
		// In the solving functions matrices are seen to be saved MATLAB-internally in
		// row-order.
		// (i.e. the matrix [1 2; 3 4] will be stored as a vector [1 3 2 4], NOT [1 2 3
		// 4]).
		for (int i = 0; i < nRows; i++) {
			final double[] line = matrix_to_copy[i];
			for (int j = 0; j < nCols; j++) {
				double val = line[j];
				distMatrixCopy[j][i] = val;
			}
		}
		return distMatrixCopy;
	}

	/**
	 * XXX Why copy dist matrix: change interface directly. I see the point that
	 * this is linearized, somehow. Changed this in Java code
	 */
	@Override
	public double Solve(final double[][] DistMatrix, final ArrayList<Integer> Assignment) {
		final int nRows = DistMatrix.length;
		final int nCols = DistMatrix[0].length;

		final double[][] distMatrixIn = copy_and_transpose(DistMatrix, nRows, nCols);
		final int[] assignment 		  = new int[nRows];
		this.cost 				   	  = 0.0;// XXX - I need to make this a class member. In C++ passed by reference

		// call solving function
		assignmentoptimal(assignment, distMatrixIn, nRows, nCols);

		Assignment.clear();// XXX - Why do we need this instead of using simply assignment-array. Do we need that at all?
		for (int r = 0; r < nRows; r++) {
			Assignment.add(assignment[r]);
		}
		return cost;
	}

	final double get_min_value(final double[] array) {
		double min = array[0];
		double val;
		for (int i = 1; i < array.length; i++) {
			if ((val = array[i]) < min) {
				min = val;
			}
		}
		return min;
	}

	@Override
	void assignmentoptimal(final int[] assignment, final double[][] distMatrix, final int nOfRows,
			final int nOfColumns) {
		boolean[] coveredColumns;
		boolean[] coveredRows;
		boolean[] starMatrix;
		boolean[] newStarMatrix;
		boolean[] primeMatrix;
		int nOfElements, minDim, row, col;

		/* initialization */
		this.cost = 0;
		Arrays.fill(assignment, -1);

		/* generate working copy of distance Matrix */ // XXX - why do we do that again after having linearized the
														// matrix above
		/* check if all matrix elements are positive */
		nOfElements = nOfRows * nOfColumns;
		/*
		 * distMatrix = new double[nOfElements]; distMatrixEnd = nOfElements;//XXX
		 * pointer magic?
		 * 
		 * for (row = 0; row<nOfElements; row++){//TODO - candidate for removal value =
		 * distMatrixIn[row]; if (value < 0)
		 * System.err.println("All matrix elements have to be non-negative.");
		 * distMatrix[row] = value; }
		 */

		/* memory allocation */ // XXX why not pre-allocate them: They all have the same size
		coveredColumns = new boolean[nOfColumns]; 	// (bool *)calloc(nOfColumns, sizeof(bool));
		coveredRows    = new boolean[nOfRows]; 		// (bool *)calloc(nOfRows, sizeof(bool));
		starMatrix     = new boolean[nOfElements];	// (bool *)calloc(nOfElements, sizeof(bool));
		primeMatrix    = new boolean[nOfElements];	// (bool *)calloc(nOfElements, sizeof(bool));
		newStarMatrix  = new boolean[nOfElements];  // (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

		minDim = nOfRows;
		/* preliminary steps */
		if (nOfRows <= nOfColumns) {
			for (row = 0; row < nOfRows; row++) {
				double minValue;
				/* find the smallest element in the row */
				double[] matrix_row = distMatrix[row];
				minValue = get_min_value(matrix_row);

				/* subtract the smallest element from each element of the row */
				for (int i = 0; i < matrix_row.length; i++) {
					matrix_row[i] -= minValue;
				}
			}

			/* Steps 1 and 2a */
			for (row = 0; row < nOfRows; row++) {
				final double[] matrix_row = distMatrix[row];
				for (col = 0; col < nOfColumns; col++) {
					if (fabs(matrix_row[col]) < DBL_EPSILON) {
						if (!coveredColumns[col]) {
							starMatrix[row + nOfRows * col] = true;
							coveredColumns[col] = true;
							break;
						}
					}
				}
			}
		} else {/* if(nOfRows > nOfColumns) */
			System.err.println("if(nOfRows > nOfColumns) - should never happen");
			// removed. All windows have the same size
		}

		/* move to step 2b */
		step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
				nOfColumns, minDim);

		/* compute cost and remove invalid assignments */
		computeassignmentcost(assignment, distMatrix, nOfRows);
	}

	@Override
	void buildassignmentvector(int[] assignment, boolean[] starMatrix, int nOfRows, int nOfColumns) {
		int row, col;

		for (row = 0; row<nOfRows; row++) {
			for (col = 0; col<nOfColumns; col++){
				if (starMatrix[row + nOfRows*col]){
					assignment[row] = col;
					break;
				}
			}
		}
	}

	@Override
	final void computeassignmentcost(final int[] assignment, final double[][] distMatrix, final int nOfRows) {
		int row, col;

		for (row = 0; row < nOfRows; row++) {
			final double[] matrix_row = distMatrix[row];
			col = assignment[row];
			if (col >= 0) {
				this.cost += matrix_row[col];
			}
		}
	}

	@Override
	void step2a(int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix,
			boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns,
			int minDim) {
		/* cover every column containing a starred zero */
		for (int col = 0; col < nOfColumns; col++) {
			final int colum_start = nOfRows * col;
			final int column_end  = colum_start + nOfRows;
			int pointer = colum_start;
			while (pointer < column_end){
				if (starMatrix[pointer]) {
					coveredColumns[col] = true;
					break;
				}
				pointer++;
			}
		}

		/* move to step 3 */ //XXX - Why is it saying Step 3, but calling step 2b?
		step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
				nOfColumns, minDim);

	}

	@Override
	void step2b(int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix,
			boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns,
			int minDim) {
		int col, nOfCoveredColumns;

		/* count covered columns */
		nOfCoveredColumns = 0;
		for (col = 0; col < nOfColumns; col++)
			if (coveredColumns[col])
				nOfCoveredColumns++;

		if (nOfCoveredColumns == minDim) {
			/* algorithm finished */
			buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
		} else {
			/* move to step 3 */
			step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
					nOfColumns, minDim);
		}
	}

	@Override
	void step3(int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix,
			boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns,
			int minDim) {
		boolean zerosFound;
		int row, col, starCol;

		zerosFound = true;
		while (zerosFound) {
			zerosFound = false;
			for (col = 0; col < nOfColumns; col++)
				if (!coveredColumns[col])
					for (row = 0; row < nOfRows; row++) {
						final double[] matrix_row = distMatrix[row];// TODO here we see a non-cache sensitive access
						if ((!coveredRows[row]) && (fabs(matrix_row[col]) < DBL_EPSILON)) {
							/* prime zero */
							primeMatrix[row + nOfRows * col] = true;

							/* find starred zero in current row */
							for (starCol = 0; starCol < nOfColumns; starCol++)
								if (starMatrix[row + nOfRows * starCol])
									break;

							if (starCol == nOfColumns) /* no starred zero found */
							{
								/* move to step 4 */
								step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns,
										coveredRows, nOfRows, nOfColumns, minDim, row, col);
								return;
							} else {
								coveredRows[row] = true;
								coveredColumns[starCol] = false;
								zerosFound = true;
								break;
							}
						}
					}
		}

		/* move to step 5 */
		step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
				nOfColumns, minDim);
	}

	@Override
	void step4(int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix,
			boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns,
			int minDim, int row, int col) {
		int n, starRow, starCol, primeRow, primeCol;
		int nOfElements = nOfRows * nOfColumns;

		/* generate temporary copy of starMatrix */
		for (n = 0; n < nOfElements; n++)
			newStarMatrix[n] = starMatrix[n];

		/* star current zero */
		newStarMatrix[row + nOfRows * col] = true;

		/* find starred zero in current column */
		starCol = col;
		for (starRow = 0; starRow < nOfRows; starRow++)
			if (starMatrix[starRow + nOfRows * starCol])
				break;

		while (starRow < nOfRows) {
			/* unstar the starred zero */
			newStarMatrix[starRow + nOfRows * starCol] = false;

			/* find primed zero in current row */
			primeRow = starRow;
			for (primeCol = 0; primeCol < nOfColumns; primeCol++)
				if (primeMatrix[primeRow + nOfRows * primeCol])
					break;

			/* star the primed zero */
			newStarMatrix[primeRow + nOfRows * primeCol] = true;

			/* find starred zero in current column */
			starCol = primeCol;
			for (starRow = 0; starRow < nOfRows; starRow++)
				if (starMatrix[starRow + nOfRows * starCol])
					break;
		}

		/* use temporary copy as new starMatrix */
		/* delete all primes, uncover all rows */
		for (n = 0; n < nOfElements; n++) {
			primeMatrix[n] = false;
			starMatrix[n] = newStarMatrix[n];
		}
		for (n = 0; n < nOfRows; n++)
			coveredRows[n] = false;

		/* move to step 2a */
		step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
				nOfColumns, minDim);

	}

	@Override
	void step5(int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix,
			boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns,
			int minDim) {
		double h, value;
		int row, col;

		/* find smallest uncovered element h */
		h = DBL_MAX;
		for (row = 0; row < nOfRows; row++) {
			final double[] matrix_line = distMatrix[row];
			if (!coveredRows[row]) {
				for (col = 0; col < nOfColumns; col++) {
					if (!coveredColumns[col]) {
						value = matrix_line[col];
						if (value < h)
							h = value;
					}
				}
			}
		}
		/* add h to each covered row */
		for (row = 0; row < nOfRows; row++) {
			final double[] matrix_line = distMatrix[row];
			if (coveredRows[row]) {
				for (col = 0; col < nOfColumns; col++) {
					matrix_line[col] += h;
				}
			}
		}
		/* subtract h from each uncovered column */
		for (col = 0; col < nOfColumns; col++) {
			if (!coveredColumns[col]) {
				for (row = 0; row < nOfRows; row++) {
					distMatrix[row][col] -= h;// None-cache friendly access
				}
			}
		}
		/* move to step 3 */
		step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows,
				nOfColumns, minDim);
	}

}

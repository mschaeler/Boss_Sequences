package boss.hungarian;

import java.util.Arrays;

/* Copyright (c) 2012 Kevin L. Stern
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * An implementation of the Hungarian algorithm for solving the assignment
 * problem. An instance of the assignment problem consists of a number of
 * workers along with a number of jobs and a cost matrix which gives the cost of
 * assigning the i'th worker to the j'th job at position (i, j). The goal is to
 * find an assignment of workers to jobs so that no job is assigned more than
 * one worker and so that no worker is assigned to more than one job in such a
 * manner so as to minimize the total cost of completing the jobs.
 * <p>
 * 
 * An assignment for a cost matrix that has more workers than jobs will
 * necessarily include unassigned workers, indicated by an assignment value of
 * -1; in no other circumstance will there be unassigned workers. Similarly, an
 * assignment for a cost matrix that has more jobs than workers will necessarily
 * include unassigned jobs; in no other circumstance will there be unassigned
 * jobs. For completeness, an assignment for a square cost matrix will give
 * exactly one unique worker to each job.
 * <p>
 * 
 * This version of the Hungarian algorithm runs in time O(n^3), where n is the
 * maximum among the number of workers and the number of jobs.
 * 
 * @author Kevin L. Stern
 */

public class HungarianKevinSternAlmpified extends Solver{
	private final double[][] buffer;
	private double[][] costMatrix;
	private final int rows, cols, dim;
	private final double[] labelByWorker, labelByJob;
	private final int[] minSlackWorkerByJob;
	private final double[] minSlackValueByJob;
	private final int[] matchJobByWorker, matchWorkerByJob;
	private final int[] parentWorkerByCommittedJob;
	private final boolean[] committedWorkers;
	
	public int solve_counter=0;
	public int phase_counter=0;

	/**
	 * Construct an instance of the algorithm.
	 * 
	 * @param costMatrix the cost matrix, where matrix[i][j] holds the cost of
	 *                   assigning worker i to job j, for all i, j. The cost matrix
	 *                   must not be irregular in the sense that all rows must be
	 *                   the same length; in addition, all entries must be
	 *                   non-infinite numbers.
	 */
	public HungarianKevinSternAlmpified(int k) {
		this.dim = k;
		this.rows = k;
		this.cols = k;
		this.buffer = new double[this.dim][this.dim];
		this.costMatrix = buffer;
		
		labelByWorker = new double[this.dim];
		labelByJob = new double[this.dim];
		minSlackWorkerByJob = new int[this.dim];
		minSlackValueByJob = new double[this.dim];
		committedWorkers = new boolean[this.dim];
		parentWorkerByCommittedJob = new int[this.dim];
		matchJobByWorker = new int[this.dim];
		matchWorkerByJob = new int[this.dim];
	}

	/**
	 * Compute an initial feasible solution by assigning zero labels to the workers
	 * and by assigning to each job a label equal to the minimum cost among its
	 * incident edges.
	 */
	protected void computeInitialFeasibleSolution() {
		for (int j = 0; j < dim; j++) {
			labelByJob[j] = Double.POSITIVE_INFINITY;
		}
		for (int w = 0; w < dim; w++) {
			for (int j = 0; j < dim; j++) {
				if (costMatrix[w][j] < labelByJob[j]) {
					labelByJob[j] = costMatrix[w][j];
				}
			}
		}
	}
	
	/**
	 * Compute an initial feasible solution by assigning zero labels to the workers
	 * and by assigning to each job a label equal to the minimum cost among its
	 * incident edges.
	 */
	protected void computeInitialFeasibleSolution(final double[] col_minima) {
		for (int w = 0; w < dim; w++) {
			for (int j = 0; j < dim; j++) {
				if (costMatrix[w][j] == col_minima[j]) {
					labelByJob[j] = costMatrix[w][j];
					if (matchJobByWorker[w] == -1 && matchWorkerByJob[j] == -1) {
						match(w, j);//greedily match them
					} 
				}
			}
		}
	}
	
	/**
	 * Execute the algorithm.
	 * 
	 * @return the minimum cost matching of workers to jobs based upon the provided
	 *         cost matrix. A matching value of -1 indicates that the corresponding
	 *         worker is unassigned.
	 */
	public double solve(final double[][] org_cost_matrix, final double threshold, final double[] col_minima) {
		solve_counter++;//For statistics only
		this.costMatrix = org_cost_matrix;
		
		Arrays.fill(labelByWorker, 0);
		Arrays.fill(labelByJob, 0);
		Arrays.fill(minSlackWorkerByJob, 0);
		Arrays.fill(minSlackValueByJob, 0);
		Arrays.fill(committedWorkers, false);
		Arrays.fill(parentWorkerByCommittedJob, 0);
		
		Arrays.fill(matchJobByWorker, -1);
		Arrays.fill(matchWorkerByJob, -1);
		
		//This is the new heuristics - FIXME
		//System.err.println("Does not work");
		//int worker = matchWorkerByJob[new_job];
		//un_match(worker, new_job);

		computeInitialFeasibleSolution(col_minima);//This is the difference.
		//greedyMatch();
		
		int w = fetchUnmatchedWorker();
		while (w < dim) {
			initializePhase(w);
			executePhase();
			w = fetchUnmatchedWorker();
		}
		//DONE - Collect result
		
		double cost = 0;
		for(w=0; w<matchJobByWorker.length;w++) {
			cost += org_cost_matrix[w][matchJobByWorker[w]];
		}
		
		return cost;
	}
	
	/**
	 * Execute the algorithm.
	 * 
	 * @return the minimum cost matching of workers to jobs based upon the provided
	 *         cost matrix. A matching value of -1 indicates that the corresponding
	 *         worker is unassigned.
	 */
	public double solve(final double[][] org_cost_matrix, double threshold) {
		//Note, we need to copy the matrix, because we modify the values in between
		for (int w = 0; w < this.dim; w++) {
			this.costMatrix[w] = Arrays.copyOf(org_cost_matrix[w], this.dim);
		}
		
		Arrays.fill(labelByWorker, 0);
		Arrays.fill(labelByJob, 0);//TODO init with positive inf
		Arrays.fill(minSlackWorkerByJob, 0);
		Arrays.fill(minSlackValueByJob, 0);
		Arrays.fill(committedWorkers, false);
		Arrays.fill(parentWorkerByCommittedJob, 0);
		
		Arrays.fill(matchJobByWorker, -1);
		Arrays.fill(matchWorkerByJob, -1);
		/*
		 * Heuristics to improve performance: Reduce rows and columns by their smallest
		 * element, compute an initial non-zero dual feasible solution and create a
		 * greedy matching from workers to jobs of the cost matrix.
		 */
		reduce();
		computeInitialFeasibleSolution();
		greedyMatch();

		int w = fetchUnmatchedWorker();
		while (w < dim) {
			initializePhase(w);
			executePhase();
			w = fetchUnmatchedWorker();
		}
		//DONE - Collect result
		
		double cost = 0;
		double cost_2 = 0;
		for(w=0; w<matchJobByWorker.length;w++) {
			cost  +=org_cost_matrix[w][matchJobByWorker[w]];
			cost_2+=org_cost_matrix[w][matchWorkerByJob[w]];
		}
		
		return cost;
	}

	/**
	 * Execute a single phase of the algorithm. A phase of the Hungarian algorithm
	 * consists of building a set of committed workers and a set of committed jobs
	 * from a root unmatched worker by following alternating unmatched/matched
	 * zero-slack edges. If an unmatched job is encountered, then an augmenting path
	 * has been found and the matching is grown. If the connected zero-slack edges
	 * have been exhausted, the labels of committed workers are increased by the
	 * minimum slack among committed workers and non-committed jobs to create more
	 * zero-slack edges (the labels of committed jobs are simultaneously decreased
	 * by the same amount in order to maintain a feasible labeling).
	 * <p>
	 * 
	 * The runtime of a single phase of the algorithm is O(n^2), where n is the
	 * dimension of the internal square cost matrix, since each edge is visited at
	 * most once and since increasing the labeling is accomplished in time O(n) by
	 * maintaining the minimum slack values among non-committed jobs. When a phase
	 * completes, the matching will have increased in size.
	 */
	protected void executePhase() {
		phase_counter++;//For statistics only
		while (true) {
			int minSlackWorker = -1, minSlackJob = -1;
			double minSlackValue = Double.POSITIVE_INFINITY;
			for (int j = 0; j < dim; j++) {
				if (parentWorkerByCommittedJob[j] == -1) {
					if (minSlackValueByJob[j] < minSlackValue) {
						minSlackValue = minSlackValueByJob[j];
						minSlackWorker = minSlackWorkerByJob[j];
						minSlackJob = j;
					}
				}
			}
			if (minSlackValue > 0) {
				updateLabeling(minSlackValue);
			}
			parentWorkerByCommittedJob[minSlackJob] = minSlackWorker;
			if (matchWorkerByJob[minSlackJob] == -1) {
				/*
				 * An augmenting path has been found.
				 */
				int committedJob = minSlackJob;
				int parentWorker = parentWorkerByCommittedJob[committedJob];
				while (true) {
					int temp = matchJobByWorker[parentWorker];
					match(parentWorker, committedJob);
					committedJob = temp;
					if (committedJob == -1) {
						break;
					}
					parentWorker = parentWorkerByCommittedJob[committedJob];
				}
				return;
			} else {
				/*
				 * Update slack values since we increased the size of the committed workers set.
				 */
				int worker = matchWorkerByJob[minSlackJob];
				committedWorkers[worker] = true;
				for (int j = 0; j < dim; j++) {
					if (parentWorkerByCommittedJob[j] == -1) {
						double slack = costMatrix[worker][j] - labelByWorker[worker] - labelByJob[j];
						if (minSlackValueByJob[j] > slack) {
							minSlackValueByJob[j] = slack;
							minSlackWorkerByJob[j] = worker;
						}
					}
				}
			}
		}
	}

	/**
	 * 
	 * @return the first unmatched worker or {@link #dim} if none.
	 */
	protected int fetchUnmatchedWorker() {
		int w;
		for (w = 0; w < dim; w++) {
			if (matchJobByWorker[w] == -1) {
				break;
			}
		}
		return w;
	}

	/**
	 * Find a valid matching by greedily selecting among zero-cost matchings. This
	 * is a heuristic to jump-start the augmentation algorithm.
	 */
	protected void greedyMatch() {
		for (int w = 0; w < dim; w++) {
			for (int j = 0; j < dim; j++) {
				if (matchJobByWorker[w] == -1 && matchWorkerByJob[j] == -1
						&& costMatrix[w][j] - labelByWorker[w] - labelByJob[j] == 0) {
					match(w, j);
				}
			}
		}
	}

	/**
	 * Initialize the next phase of the algorithm by clearing the committed workers
	 * and jobs sets and by initializing the slack arrays to the values
	 * corresponding to the specified root worker.
	 * 
	 * @param w the worker at which to root the next phase.
	 */
	protected void initializePhase(int w) {
		Arrays.fill(committedWorkers, false);
		Arrays.fill(parentWorkerByCommittedJob, -1);
		committedWorkers[w] = true;
		for (int j = 0; j < dim; j++) {
			minSlackValueByJob[j] = costMatrix[w][j] - labelByWorker[w] - labelByJob[j];
			minSlackWorkerByJob[j] = w;
		}
	}

	/**
	 * Helper method to record a matching between worker w and job j.
	 */
	protected void match(int w, int j) {
		matchJobByWorker[w] = j;
		matchWorkerByJob[j] = w;
	}
	
	/**
	 * Helper method to re-use the matching of the prior cell.
	 */
	protected void un_match(int w, int j) {
		matchJobByWorker[w] = -1;
		matchWorkerByJob[j] = -1;
	}

	/**
	 * Reduce the cost matrix by subtracting the smallest element of each row from
	 * all elements of the row as well as the smallest element of each column from
	 * all elements of the column. Note that an optimal assignment for a reduced
	 * cost matrix is optimal for the original cost matrix.
	 */
	protected void reduce(final double[] column_minima) {
		/* This is what we save
		double[] min = new double[dim];
		for (int j = 0; j < dim; j++) {
			min[j] = Double.POSITIVE_INFINITY;
		}
		for (int w = 0; w < dim; w++) {
			for (int j = 0; j < dim; j++) {
				if (costMatrix[w][j] < min[j]) {
					min[j] = costMatrix[w][j];
				}
			}
		}*/
		for (int w = 0; w < dim; w++) {
			for (int j = 0; j < dim; j++) {
				costMatrix[w][j] -= column_minima[j];//XXX here we indeed modify the matrix
			}
		}
		
		for (int w = 0; w < dim; w++) {
			double min_r = Double.POSITIVE_INFINITY;
			for (int j = 0; j < dim; j++) {
				if (costMatrix[w][j] < min_r) {
					min_r = costMatrix[w][j];
				}
			}
			for (int j = 0; j < dim; j++) {
				costMatrix[w][j] -= min_r;//XXX here we indeed modify the matrix
			}
		}
	}
	
	/**
	 * Reduce the cost matrix by subtracting the smallest element of each row from
	 * all elements of the row as well as the smallest element of each column from
	 * all elements of the column. Note that an optimal assignment for a reduced
	 * cost matrix is optimal for the original cost matrix.
	 */
	protected void reduce() {
		for (int w = 0; w < dim; w++) {
			double min = Double.POSITIVE_INFINITY;
			for (int j = 0; j < dim; j++) {
				if (costMatrix[w][j] < min) {
					min = costMatrix[w][j];
				}
			}
			for (int j = 0; j < dim; j++) {
				costMatrix[w][j] -= min;//XXX here we indeed modify the matrix
			}
		}
		double[] min = new double[dim];
		for (int j = 0; j < dim; j++) {
			min[j] = Double.POSITIVE_INFINITY;
		}
		for (int w = 0; w < dim; w++) {
			for (int j = 0; j < dim; j++) {
				if (costMatrix[w][j] < min[j]) {
					min[j] = costMatrix[w][j];
				}
			}
		}
		for (int w = 0; w < dim; w++) {
			for (int j = 0; j < dim; j++) {
				costMatrix[w][j] -= min[j];//XXX here we indeed modify the matrix
			}
		}
	}

	/**
	 * Update labels with the specified slack by adding the slack value for
	 * committed workers and by subtracting the slack value for committed jobs. In
	 * addition, update the minimum slack values appropriately.
	 */
	protected void updateLabeling(double slack) {
		for (int w = 0; w < dim; w++) {
			if (committedWorkers[w]) {
				labelByWorker[w] += slack;
			}
		}
		for (int j = 0; j < dim; j++) {
			if (parentWorkerByCommittedJob[j] != -1) {
				labelByJob[j] -= slack;
			} else {
				minSlackValueByJob[j] -= slack;
			}
		}
	}

	@Override
	public String get_name() {
		return "Hungarian Algorithm by Kevin Stern with injected results";
	}
	
	public String get_statistics() {
		String s = "#solve\t#phases";
		s+="\n"+this.solve_counter+"\t"+this.phase_counter;
		return s;
	}
}
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

public class HungarianDeep extends Solver{
	private double[][] costMatrix;
	private final int rows, cols, dim;
	private final double[] labelByWorker, labelByJob;
	private final int[] minSlackWorkerByJob;
	private final double[] minSlackValueByJob;
	public final int[] matchJobByWorker, matchWorkerByJob;
	private final int[] parentWorkerByCommittedJob;
	private final boolean[] committedWorkers;
	
	public int solve_counter=0;
	public int phase_counter=0;
	
	int start_offset;

	/**
	 * Construct an instance of the algorithm.
	 * 
	 * @param costMatrix the cost matrix, where matrix[i][j] holds the cost of
	 *                   assigning worker i to job j, for all i, j. The cost matrix
	 *                   must not be irregular in the sense that all rows must be
	 *                   the same length; in addition, all entries must be
	 *                   non-infinite numbers.
	 */
	public HungarianDeep(int k) {
		this.dim = k;
		this.rows = k;
		this.cols = k;
		//this.costMatrix = new double[this.dim][this.dim];
		
		labelByWorker = new double[this.dim];
		labelByJob = new double[this.dim];
		minSlackWorkerByJob = new int[this.dim];
		minSlackValueByJob = new double[this.dim];
		committedWorkers = new boolean[this.dim];
		parentWorkerByCommittedJob = new int[this.dim];
		matchJobByWorker = new int[this.dim];
		matchWorkerByJob = new int[this.dim];
	}
	
	void set_matrix(final double[][] matrix) {
		this.costMatrix = matrix;
	}
	
	/**
	 * Execute the algorithm.
	 * 
	 * @return the minimum cost matching of workers to jobs based upon the provided
	 *         cost matrix. A matching value of -1 indicates that the corresponding
	 *         worker is unassigned.
	 */
	public double solve(final int start_column, final double[] col_minima) {
		solve_counter++;//For statistics only
		this.start_offset = start_column;
		
		Arrays.fill(labelByWorker, 0);
		/*Arrays.fill(labelByJob, 0);
		Arrays.fill(minSlackWorkerByJob, 0);
		Arrays.fill(minSlackValueByJob, 0);
		Arrays.fill(committedWorkers, false);
		Arrays.fill(parentWorkerByCommittedJob, 0);
		*/
		Arrays.fill(matchJobByWorker, -1);
		Arrays.fill(matchWorkerByJob, -1);

		computeInitialFeasibleSolution(col_minima);//This is the difference.
		
		int w = fetchUnmatchedWorker();
		while (w < dim) {
			initializePhase(w);
			executePhase();
			w = fetchUnmatchedWorker();
		}
		//DONE - Collect result
		
		double cost = get_cost();
		return cost;
	}
	
	/**
	 * Compute an initial feasible solution by assigning zero labels to the workers
	 * and by assigning to each job a label equal to the minimum cost among its
	 * incident edges.
	 */
	protected void computeInitialFeasibleSolution(final double[] col_minima) {
		for (int w = 0; w < dim; w++) {
			for (int j = 0; j < dim; j++) {
				if (cost(w,j) == col_minima[j]) {
					labelByJob[j] = cost(w,j);
					if (matchJobByWorker[w] == -1 && matchWorkerByJob[j] == -1) {
						match(w, j);//greedily match them
					} 
				}
			}
		}
	}
	
	public final double get_cost() {
		double cost = 0;
		for(int w=0; w<matchJobByWorker.length;w++) {
			cost  +=cost(w,matchJobByWorker[w]);
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
	private void executePhase() {
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
						double slack = cost(worker,j) - labelByWorker[worker] - labelByJob[j];
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
	private int fetchUnmatchedWorker() {
		int w;
		for (w = 0; w < dim; w++) {
			if (matchJobByWorker[w] == -1) {
				break;
			}
		}
		return w;
	}

	/**
	 * Initialize the next phase of the algorithm by clearing the committed workers
	 * and jobs sets and by initializing the slack arrays to the values
	 * corresponding to the specified root worker.
	 * 
	 * @param w the worker at which to root the next phase.
	 */
	private void initializePhase(int w) {
		Arrays.fill(committedWorkers, false);
		Arrays.fill(parentWorkerByCommittedJob, -1);
		committedWorkers[w] = true;
		for (int j = 0; j < dim; j++) {
			minSlackValueByJob[j] = cost(w,j) - labelByWorker[w] - labelByJob[j];
			minSlackWorkerByJob[j] = w;
		}
	}

	/**
	 * Helper method to record a matching between worker w and job j.
	 */
	private void match(int w, int j) {
		matchJobByWorker[w] = j;
		matchWorkerByJob[j] = w;
	}

	/**
	 * Update labels with the specified slack by adding the slack value for
	 * committed workers and by subtracting the slack value for committed jobs. In
	 * addition, update the minimum slack values appropriately.
	 */
	private void updateLabeling(double slack) {
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
		return "Hungarian Algorithm by Kevin Stern";
	}
	
	@Override
	public int get_deleted_node_assigment() {
		return matchWorkerByJob[0];
	}

	@Override
	public double solve(double[][] cost_matrix, double threshold) {
		System.err.println("Not implemented");
		return 0;
	}
	public double solve(final int start_column) {
		solve_counter++;//For statistics only
		this.start_offset = start_column;
		
		Arrays.fill(labelByWorker, 0);
		Arrays.fill(labelByJob, 0);
		Arrays.fill(minSlackWorkerByJob, 0);
		Arrays.fill(minSlackValueByJob, 0);
		Arrays.fill(committedWorkers, false);
		Arrays.fill(parentWorkerByCommittedJob, 0);
		
		Arrays.fill(matchJobByWorker, -1);
		Arrays.fill(matchWorkerByJob, -1);
		
		double[] col_minima = new double[dim];
		for(int column=0;column<dim;column++) {
			double cost = cost(0, column);
			col_minima[column] = cost;
		}
		for(int line=1;line<dim;line++) {
			for(int column=0;column<dim;column++) {
				double cost = cost(line, column);
				if(cost<col_minima[column]) {
					col_minima[column] = cost;
				}
			}
		}
		
		computeInitialFeasibleSolution(col_minima);//This is the difference.
		
		int w = fetchUnmatchedWorker();
		while (w < dim) {
			initializePhase(w);
			executePhase();
			w = fetchUnmatchedWorker();
		}
		//DONE - Collect result
		
		double cost = get_cost();
		return cost;
	}
	
	final double cost(final int w, final int j) {
		return costMatrix[w][start_offset+j];
	}
	
	public double[][] get_matrix_copy(double[][] current_window, int offset){
		this.costMatrix = current_window;
		this.start_offset = offset;
		
		double[][] matrix = new double[dim][dim];
		for(int w=0;w<dim;w++) {
			for(int j=0;j<dim;j++) {
				matrix[w][j] = cost(w, j);
			}
		}
		return matrix;
	}
}
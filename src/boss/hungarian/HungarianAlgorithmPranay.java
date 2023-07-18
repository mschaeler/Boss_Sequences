package boss.hungarian;

import java.util.ArrayList;

public abstract class HungarianAlgorithmPranay extends Solver{
	public abstract double Solve(double[][] DistMatrix, ArrayList<Integer> Assignment);
	abstract void assignmentoptimal(int[] assignment, double[][] distMatrix, int nOfRows, int nOfColumns);
	abstract void buildassignmentvector(int[] assignment, boolean[] starMatrix, int nOfRows, int nOfColumns);
	abstract void computeassignmentcost(int[] assignment, double[][] distMatrix, int nOfRows);
	abstract void step2a(int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim);
	abstract void step2b(int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim);
	abstract void step3 (int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim);
	abstract void step4 (int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
	abstract void step5 (int[] assignment, double[][] distMatrix, boolean[] starMatrix, boolean[] newStarMatrix, boolean[] primeMatrix, boolean[] coveredColumns, boolean[] coveredRows, int nOfRows, int nOfColumns, int minDim);
	
	
	//TODO
	double fabs(double value) {
		return Math.abs(value);
	}
	static final double DBL_MAX 	= Double.MAX_VALUE;
	static final double DBL_EPSILON = 2.2204460492503131e-016;/* smallest such that 1.0+DBL_EPSILON != 1.0 */
	
	public static void main(String[] args) {
		HungarianAlgorithmPranay hunger = new HungarianAlgorithmPranayImplementation();
		double[][] costMatrix = { 
				  { 10, 19, 8, 15}, 
				  { 10, 18, 7, 17}, 
				  { 13, 16, 9, 14}, 
				  { 12, 19, 8, 18} 
		};
		ArrayList<Integer> assignment = new ArrayList<Integer>(10);
		//double cost = hunger.Solve(costMatrix, assignment);
		//System.out.println("Cost="+cost);
		
		//double[][] costs= {{8, 5, 9}, {4, 2, 4}, {7, 3, 8}};
		//double[][] costs= {{8, 4, 7}, {5, 2, 3}, {9, 4, 8}};
		double[][] costs= {
				{1, 1, 1, 2}
				, {3, 2, 4, 1}
				, {4, 4, 2, 4}
				, {2, 3, 3, 3}
			};
		double cost = hunger.Solve(costs, assignment);
		System.out.println("Cost="+cost);
	}
}

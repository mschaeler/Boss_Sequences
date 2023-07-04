package boss.hungarian;

public abstract class Solver {
	public final double solve(double[][] cost_matrix) {
		return solve(cost_matrix, HungarianExperiment.MAX_DIST);
	}
	public abstract double solve(double[][] cost_matrix, final double threshold);
	
	public abstract String get_name();
}

package boss.hungarian;

/**
 * Does not use Semantic of tokens, but only token ids. It may however use running windows.
 * @author b1074672
 *
 */
public class VanillaSolver extends Solver{

	@Override
	public double solve(double[][] cost_matrix, double threshold) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	@Override
	public String get_name() {
		return this.getClass().getCanonicalName();
	}
}

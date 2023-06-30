package boss.hungarian;

import java.util.Arrays;

public class HungarianTest {
	final Matrix my_matrix;
	final int matrix_dim;
	
	//asserts nxn cost matrices
	public HungarianTest(int dim){
		this.matrix_dim = dim;
		this.my_matrix  = new Matrix();//alocate buffers once
	}
	
	public void solve(double[][] cost_matrix){
		for(int i=0;i<cost_matrix.length;i++) {
			double[] line = cost_matrix[i];
			System.arraycopy(line, 0, my_matrix.cost_matrix[i], 0, matrix_dim);
		}
		solve();
	}
	
	void solve(){
		my_matrix.clear();
		System.out.println("Starting computation");
		System.out.println(my_matrix);
		
		my_matrix.find_and_subtract_row_minima();
		System.out.println("After subtracting column minima");
		System.out.println(my_matrix);
		
		my_matrix.find_and_subtract_column_minima();
		System.out.println("After subtracting row minima");
		System.out.println(my_matrix);
	}
	
	/**
	 * 
	 * @author b1074672
	 *
	 */
	class Matrix{
		/**
		 * cost_matrix[row][line]
		 */
		final double[][] cost_matrix;
		/**
		 * star_matix[row][line]
		 */
		final boolean[][] star_matix; 
				
		public Matrix() {
			this.star_matix  = new boolean[matrix_dim][matrix_dim];
			this.cost_matrix = new double[matrix_dim][matrix_dim];
		}

		public void clear() {
			for(boolean[] array : star_matix) {
				Arrays.fill(array, false);
			}
		}

		void find_and_subtract_row_minima(){
			//find minimum
			for(int row=0;row<matrix_dim;row++){//this is not cache sensitive
				double[] matrix_line = cost_matrix[0];
				double min = matrix_line[row];
				for(int line=1;line<matrix_dim;line++) {
					matrix_line = cost_matrix[line];
					if(matrix_line[row]<min) {
						min = matrix_line[row];
					}
				}
				//subtract minimum	
				for(int line=0;line<matrix_dim;line++) {
					matrix_line = cost_matrix[line];
					matrix_line[row] -= min;
				}
			}
		}
		
		void find_and_subtract_column_minima(){
			for(double[] line : cost_matrix){//for each line - this cache sensitive
				double min = line[0];
				//find minimum
				for(int i=1;i<line.length;i++) {
					if(line[i]<min) {
						min = line[i];
					}
				}
				//subtract minimum
				for(int i=0;i<line.length;i++) {
					line[i] -= min;
				}
			}
		}
		public String toString() {
			String ret = "";
			for(double[] a : cost_matrix){
				ret += Arrays.toString(a)+"\n";
			}
			return ret;
		}
	}
	
	public static void main(String[] args) {
		//double[][] costs= {{8, 5, 9}, {4, 2, 4}, {7, 3, 8}};
		double[][] costs= {
				{1, 1, 1, 2}
				, {3, 2, 4, 1}
				, {4, 4, 2, 4}
				, {2, 3, 3, 3}
			};
		HungarianTest hunger = new HungarianTest(costs.length);
		hunger.solve(costs);
	}
}

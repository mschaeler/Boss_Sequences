package boss.hungarian;

import java.util.Arrays;

public final class MatrixRingBuffer {
	final double[][] buffer;
	final double[] col_maxima;
	final int size;
	double col_sum;
	
	public MatrixRingBuffer(final int k){
		this.size = k;
		this.buffer = new double[k][k];
		this.col_maxima = new double[k];
	}
	
	/**
	 * 
	 * @param row
	 * @param column
	 * @param sim - materialized sim function
	 * @param book_1
	 * @param book_2
	 */
	void fill(final int row, final int column, final double[][] sim, final int[] book_1, final int[] book_2) {
		for(int buffer_row=0;buffer_row<size;buffer_row++) {
			final double[] current_row = buffer[buffer_row];
			final int token_book_1 = book_1[row+buffer_row];
			for(int buffer_col=0;buffer_col<size;buffer_col++) {
				final int token_book_2 = book_2[column+buffer_col];
				current_row[(column+buffer_col)%size] = -sim[token_book_1][token_book_2];
			}
		}
	}
	void update(final int row, final int start_column, final double[][] sim, final int[] book_1, final int[] book_2) {
		final int token_offset_b2 = start_column+size-1;
		final int token_book_2 = book_2[token_offset_b2];
		final int buffer_index = (start_column-1)%size;
		
		for(int buffer_row=0;buffer_row<size;buffer_row++) {
			final int token_book_1 = book_1[row+buffer_row];
			buffer[buffer_row][buffer_index] = -sim[token_book_1][token_book_2];
		}
	}
	public void update_with_bound(final int row, final int start_column, final double[][] sim, final int[] book_1, final int[] book_2) {
		final int token_offset_b2 = start_column+size-1;
		final int token_book_2 = book_2[token_offset_b2];
		final int buffer_index = (start_column-1)%size;
		
		final double old_col_max = col_maxima[buffer_index];
		double max = Double.POSITIVE_INFINITY;
		
		for(int buffer_row=0;buffer_row<size;buffer_row++) {
			final int token_book_1 = book_1[row+buffer_row];
			double neg_similarity = -sim[token_book_1][token_book_2];
			if(neg_similarity<max) {
				max = neg_similarity;
			}
			buffer[buffer_row][buffer_index] = neg_similarity;
		}
		col_sum-=old_col_max;
		col_sum+=col_maxima[buffer_index]=max;
		
		double[] temp = new double[size];
		Arrays.fill(temp, Double.POSITIVE_INFINITY);
		for(double[] arr : buffer) {
			for(int c=0;c<size;c++) {
				if(arr[c]<col_maxima[c]) {
					System.err.println("arr[c]<col_maxima[c]");
				}
				if(arr[c]<temp[c]) {
					temp[c] = arr[c];
				}
			}
		}
		double sum = sum(temp);
		if(Math.abs(sum-col_sum)>Solutions.DOUBLE_PRECISION_BOUND) {
			System.err.println("sum!=col_sum");
		}
	}
	
	public double get_sum_of_column_row_minima() {
		double row_sum = 0;
		Arrays.fill(col_maxima, Double.MAX_VALUE);
		for(int i=0;i<size;i++) {
			final double[] line = buffer[i];
			double row_min = Double.MAX_VALUE;
			for(int j=0;j<size;j++) {
				final double val = line[j];
				if(val<row_min) {
					row_min = val;
				}
				if(val<col_maxima[j]) {
					col_maxima[j] = val;
				}
			}
			row_sum += row_min;
		}
		col_sum = sum(col_maxima);
		double max_similarity = -Math.max(row_sum, col_sum);		
		
		return max_similarity;
	}
	
	public double o_k_square_bound() {
		double row_sum = 0;
		for(int i=0;i<size;i++) {
			final double[] line = buffer[i];
			double row_min = Double.MAX_VALUE;
			for(int j=0;j<size;j++) {
				final double val = line[j];
				if(val<row_min) {
					row_min = val;
				}
			}
			row_sum += row_min;
		}
		double max_similarity = -Math.max(row_sum, col_sum);		
		
		return max_similarity;
	}
	
	void compare(double[][] local_sim_matrix, final int index){		
		for(int line=0;line<size;line++) {
			for(int column=0;column<size;column++) {
				final int buffer_index = (index+column)%size;				
				if(local_sim_matrix[line][column]!=buffer[line][buffer_index]) {
					System.out.println("LSM");
					for(double[] arr : local_sim_matrix) {
						System.out.println(Arrays.toString(arr));
					}
					System.out.println(out(index));
					System.out.println();
				}
			}
		}
	}
	public String out(final int index) {
		String ret = toString();
		ret+="Matrix\n";
		for(double[] arr : buffer) {
			for(int i=0;i<size;i++) {
				ret+=arr[(index+i)%size]+"\t";
			}
			ret+="\n";
		}
		return ret;
	}
	
	public String toString() {
		String ret = "";
		ret+="Buffer+\n";
		for(double[] arr : buffer) {
			ret+=Arrays.toString(arr)+"\n";
		}
		
		return ret;
	}
	private static double sum(final double[] array) {
		double sum = 0;
		for(double d : array) {
			sum+=d;
		}
		return sum;
	}

	public void compare_deep(double[][] current_lines, int start_offset) {
		final double[][] local_similarity_matrix = new double[size][size];
		for(int line=0;line<size;line++) {
			for(int column=0;column<size;column++) {
				local_similarity_matrix[line][column] = current_lines[line][start_offset+column];
			}
		}
		compare(local_similarity_matrix, start_offset);
	}

	private int get_offset(final int column) {
		return column%size;
	}
	
	public double min(final int column) {
		final int buffer_offset = get_offset(column+size-1);

		double min = buffer[0][buffer_offset];
		for(int line=1;line<size;line++) {
			if(min>buffer[line][buffer_offset]) {
				min=buffer[line][buffer_offset];
			}
		}
		return -min;
	}

	public double max(final int column) {
		final int buffer_offset = get_offset(column);

		double max = Double.NEGATIVE_INFINITY;//TODO remove this line?
		for(double[] line : buffer) {
			if(max<line[buffer_offset]) {//similarity of the deleted token
				max=line[buffer_offset];
			}
		}
		return -max;
	}
	
	
	public double col_max(final int column) {
		final int buffer_index = get_offset(column);
		return col_maxima[buffer_index];
	}
}

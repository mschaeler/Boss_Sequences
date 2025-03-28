package wikipedia;

import java.util.ArrayList;

import boss.util.Util;

/**
 * Variant of a histogram based regressor, with fixed bucket widths 
 */
public class Regression {
	static final double[] oph_regression_= {
			0.44773181939258166		//0.1
			,0.4676540166002126		//0.15
			,0.48757621380784355	//0.2
			,0.5074984110154744		//0.25
			,0.5274206082231054		//0.3
			,0.5473428054307363		//0.35
			,0.5672650026383672		//0.4
			,0.5871871998459982		//0.45
			,0.6071093970536291		//0.5
			,0.6270315942612601		//0.55
			,0.646953791468891		//0.6
			,0.6668759886765219		//0.65
			,0.6867981858841528		//0.7
			,0.7067203830917838		//0.75
			,0.7266425802994148		//0.8
			,0.7465647775070456		//0.85
			,0.7664869747146765		//0.9
			,0.7864091719223074		//0.95
			,0.8063313691299384		//1.0
	};
	
	static final double min_similarity = 0;
	static final double max_similarity = 1;
	
	static final int default_num_buckets = 20;
	
	final int num_buckets;
	private final double[] predictions;
	private final Bucket[] my_buckets;
	
	final double[] from;
	final double[] to;
	
	public Regression(int _num_buckets) {
		this.num_buckets = _num_buckets;
		this.predictions = new double[num_buckets];
		this.my_buckets  = new Bucket[num_buckets];
		this.from 		 = new double[num_buckets];
		this.to 		 = new double[num_buckets];
		
		final double bucket_width = (max_similarity-min_similarity) / (double) this.num_buckets;
		for(int i=0;i<num_buckets;i++) {
			double _from = bucket_width*(double)i;
			this.from[i] = _from;
			
			double _to = bucket_width*(double)(i+1);
			this.to[i] = _to;
		}
		
	}
	
	public Regression() {
		this(Regression.default_num_buckets);
	}
		
	public void fit(double[] X, double[] Y) {
		if(X.length!=X.length) {
			System.err.println("X.length!=X.length");
			return;
		}
		
		init_buckets();
		
		for(int i=0;i<X.length;i++) {
			insert(X[i],Y[i]);
		}
		
		for(Bucket b : this.my_buckets) {
			b.fit();
		}
	}
	
	private void insert(final double x, final double y) {
		int bucket = get_bucket(x);
		this.my_buckets[bucket].insert(x,y);
	}

	private void init_buckets() {
		for(int i=0;i<this.num_buckets;i++) {
			this.my_buckets[i] = new Bucket();
		}
	}

	final int get_bucket(final double value) {
		double temp = value*(double)num_buckets;
		int bucket = (int) temp;
		if(temp<0) {
			bucket = 0;
		}
		if(temp>num_buckets-1) {
			bucket = num_buckets-1;
		}
		return bucket;
	}
	
	public double predict(double x) {
		int bucket = get_bucket(x);
		
		double prediction = this.predictions[bucket];
		return prediction;
	}
	
	public double[] predict(double[] X) {
		double[] Y = new double[X.length];
		for(int i=0;i<X.length;i++) {
			double x = X[i];
			Y[i] = predict(x);
		}
		return Y;
	}
	
	/**
	 * Root mean squared error of this bucket
	 * @return
	 */
	double rmse() {
		double mse = 0.0d;
		
		for(Bucket b : my_buckets) {
			for(int i=0;i<b.my_X.size();i++) {
				double x = b.my_prediction;
				double y = b.my_Y.get(i);
				mse += (x-y)*(x-y);
			}
		}
		
		mse /= (double) this.size();		
		double rmse = Math.sqrt(mse);
		
		return rmse;
	}
	
	/**
	 * Number of (x,y) pairs over all buckets
	 * @return
	 */
	private double size() {
		int size = 0;
		for(Bucket b : this.my_buckets) {
			size += b.size();
		}
		return size;
	}

	class Bucket{
		ArrayList<Double> my_X = new ArrayList<Double>();
		ArrayList<Double> my_Y = new ArrayList<Double>();
		Double my_rmse 	= null;
		Double my_min_y = null;
		Double my_max_y = null;
		
		double my_prediction = -1;
		
		void insert(double x, double y) {
			my_X.add(x);
			my_Y.add(y);
		}
		
		void fit() {
			double sum = Util.sum(Util.toPrimitive(my_Y));
			double count = my_Y.size();
			this.my_prediction = sum / count;//TODO square?
		}

		public int size() {
			return my_X.size();
		}
		
		/**
		 * Root mean squared error of this bucket
		 * @return
		 */
		double rmse() {
			if(my_rmse!=null) {
				return my_rmse.doubleValue();
			}
			double mse = 0.0d;
			for(int i=0;i<my_X.size();i++) {
				double x = this.my_prediction;
				double y = this.my_Y.get(i);
				mse += (x-y)*(x-y);
			}
			mse /= (double)this.size();
			
			double rmse = Math.sqrt(mse);
			my_rmse = rmse;
			
			return rmse;
		}
		
		double min_y() {
			if(my_min_y!=null) {
				return my_min_y.doubleValue();
			}
			double min = Double.POSITIVE_INFINITY;
			for(double y : this.my_Y) {
				if(y<min) {
					min = y;
				}
			}
			
			my_min_y = min;
			return min;
		}
		double max_y() {
			if(my_max_y!=null) {
				return my_max_y.doubleValue();
			}
			double max = Double.NEGATIVE_INFINITY;
			for(double y : this.my_Y) {
				if(y>max) {
					max = y;
				}
			}
			
			my_max_y = max;
			return max;
		}
	}
	
	public String toString() {		
		StringBuffer sb = new StringBuffer();
		String header = "x_min\tx_max\ty_pred\t|B|\trmse\tmin_y\tmax_y";
		
		sb.append("Regression rmse=\t"+this.rmse()+"\n");
		sb.append(header+"\n");
		for(int i=0;i<this.num_buckets;i++) {
			//sb.append("x in [");
			sb.append(String.format( "%.2f", this.from[i]));
			sb.append("\t");
			sb.append(String.format( "%.2f", this.to[i]));
			sb.append("\t");
			sb.append(this.my_buckets[i].my_prediction);
			sb.append("\t");
			sb.append(this.my_buckets[i].size());
			sb.append("\t");
			sb.append(this.my_buckets[i].rmse());
			sb.append("\t");
			sb.append(this.my_buckets[i].min_y());
			sb.append("\t");
			sb.append(this.my_buckets[i].max_y());
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
	public static void main(String args[]) {
		Regression reg = new Regression();
		reg.predict(0.1);
	}

	/**
	 * 
	 * @param matrix_1
	 * @param matrix_2
	 * @return
	 */
	public static Regression fit(double[][] matrix_1, double[][] matrix_2) {
		double[] X = flatten(matrix_1);
		double[] Y = flatten(matrix_2);
		Regression reg = new Regression();
		reg.fit(X, Y);
		
		return reg;
	}

	private static final double[] flatten(final double[][] matrix) {
		final int size = matrix[0].length*matrix.length;
		final double[] ret = new double[size];
		int i = 0;
		
		for(final double[] arr : matrix) {
			for(final double val : arr) {
				ret[i++] = val;
			}
		}
		
		return ret;
	}

	public double[] filter_strength(double[] min_thresholds) {
		double size = this.size();
		double[] result = new double[min_thresholds.length];
		
		for(int i=0;i<min_thresholds.length;i++) {
			double y_min_value = min_thresholds[i];
			//Get the first bucket having such a min value, thats the first one we need
			int first_bucket = 0;
			for(;first_bucket<this.num_buckets;first_bucket++) {
				double my_y_min = this.my_buckets[first_bucket].min_y(); 
				if(my_y_min==Double.POSITIVE_INFINITY) {
					continue;
				}
				if(my_y_min>=y_min_value) {
					break;
				}
			}
			double cummulative_size = 0;
			for(int b=first_bucket;b<this.num_buckets;b++) {
				cummulative_size+=this.my_buckets[b].size();
			}
			double fraction = cummulative_size / size;
			result[i] = fraction;
		}
		return result;
	}

	public double[][] count(double[] bert_min_thresholds) {
		double[][] counts = new double[2][bert_min_thresholds.length];
		for(Bucket b : my_buckets) {
			for(int i=0;i<bert_min_thresholds.length;i++) {
				if(b.my_prediction>=bert_min_thresholds[i]) {
					counts[0][i] +=b.size();
					for(double val : b.my_Y) {
						if(val>=bert_min_thresholds[i]) {
							counts[1][i] ++;
						}
					}
				}	
			}
		}
		return counts;
	}
}

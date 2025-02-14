package boss.util;

import java.util.ArrayList;
import java.util.HashSet;

public class Util {

	public static final long seed = 1234567;

	public static void double_array_to_byte_array(double[] line, byte[] bUFFER) {
		// TODO Auto-generated method stub
		
	}
	
	public static void intToByteArray(final int VALUE, final byte[] buffer) {
		buffer[0] = (byte) ((VALUE >> 24) & 0xFF);
		buffer[1] = (byte) ((VALUE >> 16) & 0xFF);
		buffer[2] = (byte) ((VALUE >> 8) & 0xFF);
		buffer[3] = (byte) (VALUE & 0xFF);
	}

	public static void out_end_time(double start) {
		System.out.println("[Done] in "+(System.currentTimeMillis()-start)+" ms");
	}
	
	public static int byteArrayToInt(final byte[] BUFFER) {
		int value;
		value = BUFFER[3] & 0xFF 
				| (BUFFER[2] & 0xFF) << 8 
				| (BUFFER[1] & 0xFF) << 16
				| (BUFFER[0] & 0xFF) << 24
		;
		return value;
	}

	public static String[] remove(String[] tokens, HashSet<String> stop_words) {
		ArrayList<String> temp = new ArrayList<String>();
		for(String s : tokens) {
			s=s.trim();
			if(!stop_words.contains(s)) {
				temp.add(s);
			}
		}
		String[] ret = new String[temp.size()];
		for(int i=0;i<temp.size();i++) {
			ret[i] = temp.get(i);
		}
		return ret;
	}
	public static String outTSV(double[][] matrix) {
		StringBuffer buffer = new StringBuffer(matrix.length*matrix[0].length*5);
		for(double[] array : matrix) {
			buffer.append(outTSV(array));
			buffer.append("\n");
		}
		return buffer.toString();
	}
	public static String outTSV(double[] array) {
		if (array == null)
			return "Null array";

		StringBuffer buffer = new StringBuffer(array.length*5);
		for (int index = 0; index < array.length - 1; index++) {
			buffer.append(array[index] + "\t ");
		}
		buffer.append(array[array.length - 1] + "\t");

		return buffer.toString();
	}
	public static String outTSV(int[] array) {
		if (array == null)
			return "Null array";

		StringBuffer buffer = new StringBuffer(array.length*5);
		for (int index = 0; index < array.length - 1; index++) {
			buffer.append(array[index] + "\t ");
		}
		buffer.append(array[array.length - 1] + "\t");

		return buffer.toString();
	}
	public static String outTSV(String[] array) {
		if (array == null)
			return "Null array";

		StringBuffer buffer = new StringBuffer(array.length*5);
		for (int index = 0; index < array.length - 1; index++) {
			buffer.append(array[index] + "\t ");
		}
		buffer.append(array[array.length - 1] + "\t");

		return buffer.toString();
	}
	public static String outTSV(ArrayList<String> array) {
		if (array == null)
			return "Null array";

		StringBuffer buffer = new StringBuffer(array.size()*5);
		for (int index = 0; index < array.size() - 1; index++) {
			buffer.append(array.get(index) + "\t ");
		}
		buffer.append(array.get(array.size() - 1) + "\t");

		return buffer.toString();
	}

	public static boolean not_in(int val, int[] arr) {
		for(int i : arr) {
			if(i==val) {
				return false;
			}
		}
		return true;
	}

	public static double max(double[] arr) {
		double max = Double.NEGATIVE_INFINITY;
		for(double v : arr) {
			if(v>max) {
				max = v;
			}
		}
		return max;
	}

	public static double max_col(double[][] matrix, int col) {
		double max = Double.NEGATIVE_INFINITY;
		for(double[] arr : matrix) {
			double v = arr[col]; 
			if(v>max) {
				max = v;
			}
		}
		return max;
	}

	public static double max(double[] arr, int from, int to) {
		double max = Double.NEGATIVE_INFINITY;
		for(int i=from;i<=to;i++) {
			double v = arr[i];
			if(v>max) {
				max = v;
			}
		}
		return max;
	}
	
	public static double max_col(double[][] matrix, int col, int from, int to) {
		double max = Double.NEGATIVE_INFINITY;
		for(int i=from;i<=to;i++) {
			double[] arr = matrix[i];
			double v = arr[col]; 
			if(v>max) {
				max = v;
			}
		}
		return max;
	}

	public static double[] toPrimitive(ArrayList<Double> list) {
		double[] arr = new double[list.size()];
		int iter = 0;
		for(Double d : list) {
			arr[iter++] = d.doubleValue();
		}
		return arr;
	}

	public static int sum(int[] arr) {
		int sum = 0;
		for(int i : arr) {
			sum+=i;
		}
		return sum;
	}
	
	public static double sum(double[] arr) {
		double sum = 0;
		for(double i : arr) {
			sum+=i;
		}
		return sum;
	}

	public static double avg(double[] arr) {
		double sum = sum(arr);
		return sum/(double)arr.length;
	}

	public static int[] toPrimitive(HashSet<Integer> query_ids) {
		int[] ret = new int[query_ids.size()];
		int i=0;
		for(int val : query_ids) {
			ret[i++] = val;
		}
		return ret;
	}

	public static int[] to_array(int k) {
		int[] temp = {k};
		return temp;
	}

}

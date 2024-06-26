package boss.util;

import java.util.ArrayList;
import java.util.HashSet;

public class Util {

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
}

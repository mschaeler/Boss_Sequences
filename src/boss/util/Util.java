package boss.util;

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
}
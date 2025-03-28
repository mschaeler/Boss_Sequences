package soundex;

import org.apache.commons.codec.language.ColognePhonetic;
import org.apache.commons.codec.language.Soundex;

public class PhoneticSimilarity {
	/**
	 * ColognePhonetic
	 */
	static ColognePhonetic c_p = new ColognePhonetic();
	
	static Soundex soundex = new Soundex();
	static final int soundex_length = 4;
	static final double UNDEFINED = -1.0d;
	
	static char[] soundex_encode(String s) {
		s = s.replace('ä', 'a');
		return soundex.encode(s).toCharArray();
	}
	static char[] cologne_phonetic_encode(String s) {
		return c_p.encode(s).toCharArray();
	}
	
	static double soundex_similarity(String s_1, String s_2) {
		return soundex_similarity(soundex_encode(s_1),soundex_encode(s_2));
	}
	
	static double soundex_similarity(char[] s_1, char[] s_2) {
		if(s_1.length!=soundex_length) {
			System.err.println("s_1.length!=soundex_length");
			return UNDEFINED;
		}
		if(s_1.length!=soundex_length) {
			System.err.println("s_1.length!=soundex_length");
			return UNDEFINED;
		}
		return similarity(s_1, s_2, soundex_length);
	}
	
	private static double similarity(final char[] s_1, final char[] s_2, final int length) {
		double sim = 0;
		for(int i=0;i<length;i++) {
			if(s_1[i]==s_2[i]) {
				sim++;
			}
		}
		return sim / ((double) length);
	}
	
	static void out(String s) {
		System.out.print(s+"\t");
		System.out.print(soundex_encode(s));
		System.out.print("\t");
		System.out.println(cologne_phonetic_encode(s));
	}
	
	public static void main(String[] args) {
		out("Wikipedia");
		out("Lee");
		out("Britney");
		out("bewährten");
		out("Spears");
		out("Superzicke");
		
		out("Ahasveros");
		out("Xerxes");
		out("Ahasveros");
		out("Xerxes");
		out("Ahasveros");
		out("Ahasuerus");
		out("Ahasuerus");
	}
}

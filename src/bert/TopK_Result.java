package bert;

import java.util.HashSet;

public class TopK_Result implements Comparable<TopK_Result>{
	public final int offset_susp;
	public final int offset_src;
	public final double score;
	
	/**
	 * 
	 * @param offset_susp
	 * @param offset_src
	 * @param score
	 */
	public TopK_Result(int offset_susp, int offset_src, double score) {
		this.offset_susp = offset_susp;
		this.offset_src = offset_src;
		this.score = score;
	}
	
	@Override
	public int compareTo(TopK_Result arg0) {
		return Double.compare(score, arg0.score);
	}
	
	public String toString(){
		return "("+offset_susp+","+offset_src+")->"+score;
	}

	public void out(SentenceEmbedding[] pair, int[] g_t) {
		System.out.print("score="+score);
		if(g_t[0]<=offset_susp && offset_susp<=g_t[1] && g_t[2]<=offset_src && offset_src<=g_t[3]){
			System.out.println();
		}else{
			System.out.println(" not in ground truth");
		}
		System.out.println("susp("+offset_susp+")="+pair[0].sentences.get(offset_susp));
		System.out.println("src("+offset_src+")="+pair[1].sentences.get(offset_src));
	}

	public static void out(TopK_Result[] res) {
		String ret = "";
		for(TopK_Result r : res){
			ret+=r+" ";
		}
		System.out.println(ret);
	}

	public static int overlap(TopK_Result[] res, TopK_Result[] ground_truth) {
		HashSet<Integer> g_t = new HashSet<Integer>();
		for(TopK_Result tr : ground_truth) {
			g_t.add(tr.offset_src);
		}
		int count = 0;
		for(TopK_Result tr : res) {
			if(g_t.contains(tr.offset_src)) {
				count++;
			}
		}
		return count;
	}
}

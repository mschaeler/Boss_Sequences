package bert;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import boss.hungarian.Solutions;
import boss.util.Util;
import pan.PanMetrics;

public class SentenceEmbedding {
	static int i_=17;
	static int[][] ground_truth_offsets = {
	{17, 21, 7,13}			//0
	,{15, 19, 506, 510}		//1
	,{44, 60, 29, 45}		//2
	,{89, 90, 70, 70}		//3
	,{61, 79, 99, 114}		//4
	,{30, 30, 78, 78}		//5
	,{127,129,11, 13}		//6
	,{27, 29, 67, 68}		//7
	,{45, 48, 30, 34}		//8	
	,{57, 57, 154, 155}		//9	
	,{79, 94, 182, 197}		//10	
	,{47, 61, 67, 81}		//11	
	,{6, 33, 80, 99}		//12
	,{127,128,12, 13}		//13	
	,{26, 28, 78, 80}		//14	
	,{105,106, 1, 2}		//15	
	,{127,128,12, 13}		//16
	,{30, 30,418, 418}		//16
};
	
	static final String path_pan_src = "./src_python/data/pan11/sentences/src/";
	static final String path_pan_susp = "./src_python/data/pan11/sentences/susp/";
	
	public String name;
	ArrayList<String> sentences;
	ArrayList<double[]> vectors;
	String text = "";
	
	public SentenceEmbedding(String path, String file_name){
		name = path+file_name;
		sentences= load_sentences(path, file_name);
		vectors  = load_sentence_embeddings(path, file_name);
		System.out.println(this);
		for(String s : sentences) {
			text+=s+" ";
		}
	}
	
	public SentenceEmbedding(String path, String file_name, ArrayList<String> sents) {
		name = path+file_name;
		sentences= sents;
		vectors  = load_sentence_embeddings(path, file_name);
		System.out.println(this);
		for(String s : sentences) {
			text+=s+" ";
		}
		if(sentences.size()!=vectors.size()) {
			System.err.println("sentences.size()!=vectors.size() :"+sentences.size()+" "+vectors.size());
		}
	}

	public String out_plag_passage(int start, int length) {
		return this.text.substring(start, start+length);
	}
	
	ArrayList<String> load_sentences(String path, String file_name){
		BufferedReader br;
		File f = new File(path+file_name+".txt");
		ArrayList<String> sentences = new ArrayList<String>(1000);
		
		try {
			br = new BufferedReader(new FileReader(f));
			String line;
		
	        while ((line = br.readLine()) != null) {
	        	sentences.add(line);
	        	//System.out.println(line);
	        }
	        br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return sentences;
	}
	ArrayList<double[]> load_sentence_embeddings(String path, String file_name){
		BufferedReader br;
		File f = new File(path+file_name+".vec");
		ArrayList<double[]> embedings = new ArrayList<double[]>(1000);
		
		try {
			br = new BufferedReader(new FileReader(f));
			String line;
		
	        while ((line = br.readLine()) != null) {
	        	//System.out.println(line);
	        	String[] tokens = line.split(" ");
	        	double[] arr = new double[tokens.length];
	        	for(int i=0;i<arr.length;i++) {
	        		arr[i] = Double.parseDouble(tokens[i]);
	        	}
	        	normalize(arr);
	        	embedings.add(arr);
	        }
	        br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return embedings;
	}
	
	private void normalize(double[] arr) {
		double length = 0;
		for(double v : arr) {
			length += (v*v);
		}
		length = Math.sqrt(length);
		for(int dim=0;dim<arr.length;dim++) {
			arr[dim] /= length;
		}
	}

	public String toString(){
		return this.name+"\n"+sentences.get(0)+" "+vectors.get(0)[0]+" "+vectors.get(0)[1]+" "+vectors.get(0)[2]+"...";//TODO
	}
	
	static ArrayList<SentenceEmbedding[]> load_pairs() {
		ArrayList<SentenceEmbedding[]> all_pairs = new ArrayList<SentenceEmbedding[]>();
		for(String[] names : pan.Data.plagiats) {
			SentenceEmbedding[] pair = new SentenceEmbedding[2];
			pair[0] = new SentenceEmbedding(path_pan_susp, names[0]);
			pair[1] = new SentenceEmbedding(path_pan_src, names[1]);
			all_pairs.add(pair);
		}
		return all_pairs; 
	}
	
	static double[][] get_matrix(SentenceEmbedding susp, SentenceEmbedding src){
		System.out.print("get_matrix() "+susp.name+" "+src.name+" ");
		double start = System.currentTimeMillis();
		double[][] matrix = new double[susp.size()][src.size()];
		for(int line=0;line<susp.size();line++) {
			double[] vec_susp = susp.get_vec(line); 
			for(int column=0;column<src.size();column++) {
				double[] vec_src = src.get_vec(column);	
				double sim = Solutions.cosine_similarity(vec_susp, vec_src);
				matrix[line][column] = sim;
			}
		}
		double stop = System.currentTimeMillis();
		System.out.println(" in "+(stop-start)+" ms");
		
		return matrix;
	}
	
	private double[] get_vec(int index) {
		return vectors.get(index);
	}

	private int size() {
		return vectors.size();
	}

	static int[] show_plag_passages(SentenceEmbedding[] pair, int index) {
		//Get index in boss.Data
		//int index = get_index(pair);
		int[] offsets = plag_passages(pair[0],pair[1],index);
		return offsets;
	}
	
	private static int[] plag_passages(SentenceEmbedding se_susp, SentenceEmbedding se_src, int index) {
		int[] char_offsets = pan.Data.offsets[index];
		int begin_sups = -1;
		int end_susp = -1;
		int begin_src = -1;
		int end_src = -1;
		{
			System.out.println(se_susp.out_plag_passage(char_offsets[0], char_offsets[1]-char_offsets[0]));
			int line=0;
			int size = 0;
			//Find begin susp
			for(;line<se_susp.size();line++) {
				size+=se_susp.sentences.get(line).length();
				if(size>char_offsets[0]) {
					begin_sups = line;
					break;//found
				}
			}
			//Find end susp
			for(;line<se_susp.size();line++) {
				size+=se_susp.sentences.get(line).length();
				if(size>=char_offsets[1]) {
					end_susp = line;
					break;//found
				}
			}
		}
		{
			System.out.println(se_src.out_plag_passage(char_offsets[2], char_offsets[3]-char_offsets[2]));
			int line=0;
			int size = 0;
			//Find begin src
			for(;line<se_src.size();line++) {
				size+=se_src.sentences.get(line).length();
				if(size>char_offsets[2]) {
					begin_src = line;
					break;//found
				}
			}
			//Find end susp
			for(;line<se_src.size();line++) {
				size+=se_src.sentences.get(line).length();
				if(size>=char_offsets[3]) {
					end_src = line;
					break;//found
				}
			}
		}
		int[] temp = {begin_sups,end_susp,begin_src,end_src};
		System.out.println("***********************************************");
		System.out.println("Plag at susp "+index);
		for(int i=temp[0]-1;i<=temp[1]+1;i++) {
			System.out.println(i+" "+se_susp.sentences.get(i));
		}
		System.out.println("------------------------------------------------");
		System.out.println("Plag at src "+index);
		for(int i=Math.max(0, temp[2]-1);i<=temp[3]+1;i++) {
			System.out.println(i+" "+se_src.sentences.get(i));
		}
		//System.out.println("***********************************************");
		System.out.println(Arrays.toString(temp));
		return temp;
	}

	private static int get_index(SentenceEmbedding[] pair) {
		for(int i=0;i<pan.Data.plagiats.length;i++) {
			String[] names = pan.Data.plagiats[i];
			if(pair[0].name.endsWith(names[0])) {
				if(pair[1].name.endsWith(names[1])) {
					return i;
				}
			}
			
		}
		System.err.println("get_index() Index not found");
		return -1;
	}
	
	static void show_plags(int index, ArrayList<SentenceEmbedding[]> pairs) {
		SentenceEmbedding[] pair = pairs.get(i_);
		int[] offset = show_plag_passages(pair, i_);
		ArrayList<int[]> offsets = new ArrayList<int[]>();
		offsets.add(offset);
		double[][] matrix = get_matrix(pair[0],pair[1]);
		
		
		int[] g_t = ground_truth_offsets[i_];
		System.out.println(Arrays.toString(g_t));
		for(int col=g_t[2]-1;col<=g_t[3]+1;col++) {
			System.out.print("\t"+col);
		}
		System.out.println();
		for(int line=g_t[0]-1;line<=g_t[1]+1;line++) {
			System.out.print("l="+line+"\t");
			for(int col=Math.max(0,g_t[2]-1);col<=g_t[3]+1;col++) {
				System.out.print(matrix[line][col]+"\t");
			}
			System.out.print(":\t"+Util.max(matrix[line],g_t[2],g_t[3]));
			System.out.print("\t:\t"+Util.max(matrix[line]));
			System.out.println();
		}
		System.out.println();
		for(int col=g_t[2]-1;col<=g_t[3]+1;col++) {
			System.out.print("\t"+Util.max_col(matrix, col));
		}
		System.out.println();
		for(int col=g_t[2]-1;col<=g_t[3]+1;col++) {
			System.out.print("\t"+Util.max_col(matrix, col,g_t[0],g_t[1]));
		}
		System.out.println();
	}
	
	static void out_top_sentence_pairs(ArrayList<SentenceEmbedding[]> pairs){
		int index = 0;
		{
			int num_top_k = 2*Math.max(ground_truth_offsets[index][1]-ground_truth_offsets[index][0]+1,ground_truth_offsets[index][3]-ground_truth_offsets[index][2]+1);
			TopK_Result[] res = new TopK_Result[num_top_k];
			for(int i=0;i<num_top_k;i++) {
				res[i] = new TopK_Result(-1, -1, -1);
			}
			SentenceEmbedding[] pair = pairs.get(index);
			double[][] matrix = get_matrix(pair[0],pair[1]);
			for(int line = 0;line<matrix.length;line++) {
				for(int col=0;col<matrix[0].length;col++) {
					double my_score = matrix[line][col];
					if(res[0].score<my_score) {
						res[0] = new TopK_Result(line,col,my_score);
						Arrays.sort(res);
					}
				}
			}
			System.out.println("*******************************************");
			System.out.println(index+" "+pair[0].name+" vs. "+pair[1].name);
			int[] g_t = ground_truth_offsets[index];
			System.out.println(Arrays.toString(g_t));
			for(int i = 1;i<=res.length;i++) {
				System.out.println();
				TopK_Result tkr = res[res.length-i];
				if(num_top_k/2<i-1) {
					System.out.print("Below cutoff ");
				}
				System.out.println("***Top "+(i));
				tkr.out(pair, g_t);
			}
		}
	}

	public static void main(String[] args) {
		//new SentenceEmbedding(path_pan_src, "00732");
		ArrayList<double[][]> matrices = new ArrayList<double[][]>();
		ArrayList<SentenceEmbedding[]> pairs = load_pairs();
		//show_plags(i,pairs);
		out_top_sentence_pairs(pairs);
		System.exit(0);
		
		
		ArrayList<int[]> offsets = new ArrayList<int[]>();
		for(int i=0;i<pairs.size();i++) {
			SentenceEmbedding[] pair = pairs.get(i);
			int[] offset = show_plag_passages(pair, i);
			offsets.add(offset);
			double[][] matrix = get_matrix(pair[0],pair[1]);
			matrices.add(matrix);
			//System.out.println(Util.outTSV(matrix));
		}
		/*for(int[] offset : offsets) {
			System.out.println(Util.outTSV(offset));
		}*/
		PanMetrics.run_sentence_embedding(matrices, ground_truth_offsets, pairs);
	}
	
}

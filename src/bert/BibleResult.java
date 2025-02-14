package bert;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.commons.math4.legacy.stat.correlation.PearsonsCorrelation;
import boss.util.Config;
import boss.util.Util;

/**
 * Contains all results of a query document to a set of corpus documents. Typically, we use Luther Bible as query and all other versions as corpus.
 * The results contain (1) The mapping of query paragraph to their 1-NN most similar paragraph per corpus document in all_indexes[][], (2) the corresponding similarity values, (3) the linearized matrices
 * @author b1074672
 *
 */
public class BibleResult {
	static final String folder = "./results/bible_correctness/";
	
	/**
	 * window size
	 */
	final int k;
	/**
	 * e.g., SeDA, Jaccard, ...
	 */
	final String aprraoch;
	/**
	 * contains highest average similarity value to query paragraph from all corpus docs [paragraph_id][corpus_doc_id]
	 */
	final double[][] all_sims;
	/**
	 * contains *index of* highest average similarity value to query paragraph from all corpus docs [paragraph_id][corpus_doc_id]
	 */
	int[][] all_indexes;
	/**
	 * All similarity values for all cells and documents in the corpus (is huge)
	 */
	final double[] cell_similarities;
	
	BibleResult(int _k, String _approach, double[][] _all_sims, int[][] _all_indexes, double[] _cell_similarities){
		this.k = _k;
		this.aprraoch 		   = _approach;
		this.all_sims 		   = _all_sims;
		this.all_indexes 	   = _all_indexes;
		this.cell_similarities = _cell_similarities;
	}
	
	public BibleResult(int _k, String _approach, ArrayList<double[]> _all_sims, ArrayList<int[]> _all_indexes, ArrayList<Double> _cell_similarities) {
		this.k = _k;
		this.aprraoch 		   = _approach;
		{
			int i = 0;
			this.all_sims = new double[_all_sims.size()][];
			for(double[] arr : _all_sims) {
				this.all_sims[i++] = arr;
			}
		}
		{
			int i = 0;
			this.all_indexes = new int[_all_indexes.size()][];
			for(int[] arr : _all_indexes) {
				this.all_indexes[i++] = arr;
			}
		}
		{
			int i = 0;
			this.cell_similarities = new double[_cell_similarities.size()];
			for(double d : _cell_similarities) {
				this.cell_similarities[i++] = d;
			}
		}
	}

	/**
	 * Writes this object to a file
	 */
	public void to_file() {
		File dir = new File(folder);
		if(!dir.exists()) {
			dir.mkdir();
		}
		//(1) write all_sims
		try(BufferedWriter writer = new BufferedWriter(new FileWriter(folder+"sim_"+aprraoch+"_"+k+".txt"))){
			for(double[] arr : this.all_sims) {
				writer.write(Util.outTSV(arr));
				writer.write("\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		//(2) write all_indexes
		try(BufferedWriter writer = new BufferedWriter(new FileWriter(folder+"idx_"+aprraoch+"_"+k+".txt"))){
			for(int[] arr : this.all_indexes) {
				writer.write(Util.outTSV(arr));
				writer.write("\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		//(3) write cell_similarities
		try(BufferedWriter writer = new BufferedWriter(new FileWriter(folder+"all_"+aprraoch+"_"+k+".txt"))){
			for(double d : this.cell_similarities) {
				writer.write(Double.toString(d));
				writer.write("\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	static HashMap<Integer, BibleResult> load(String approach){
		HashMap<Integer, BibleResult> res = new HashMap<Integer, BibleResult>(Config.k_s.length);
		for(int k : Config.k_s) {
			res.put(k, load(approach, k));
		}
		return res;
	}
	private static BibleResult load(String approach, int k){
		System.out.print("load("+approach+","+k+")");
		double start = System.currentTimeMillis();
		
		File f_all_sims 		 = new File(folder+"sim_"+approach+"_"+k+".txt");
		if(!f_all_sims.exists()) {
			System.err.println(f_all_sims+" does not exist");
			return null;
		}
		File f_all_indexes 		 = new File(folder+"idx_"+approach+"_"+k+".txt");
		if(!f_all_indexes.exists()) {
			System.err.println(f_all_indexes+" does not exist");
			return null;
		}
		File f_cell_similarities = new File(folder+"all_"+approach+"_"+k+".txt");
		if(!f_cell_similarities.exists()) {
			System.err.println(f_cell_similarities+" does not exist");
			return null;
		}
		
		//(1) read all_sims
		ArrayList<double[]> all_sims = new ArrayList<double[]>(200);
		try(BufferedReader reader = new BufferedReader(new FileReader(f_all_sims))){
			String line;
			    
		    while ((line = reader.readLine()) != null) {
		        String[] tokens = line.split("\t");
		        double[] values = new double[tokens.length];
		        for(int i=0;i<tokens.length;i++) {
		        	values[i] = Double.parseDouble(tokens[i]);
		        }
		        all_sims.add(values);
		    }
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//(2) read all_indexes
		ArrayList<int[]> all_indexes = new ArrayList<int[]>(200);
		try(BufferedReader reader = new BufferedReader(new FileReader(f_all_indexes))){
			String line;
			    
		    while ((line = reader.readLine()) != null) {
		        String[] tokens = line.split("\t");
		        int[] indexes= new int[tokens.length];
		        for(int i=0;i<tokens.length;i++) {
		        	indexes[i] = Integer.parseInt(tokens[i].trim());
		        }
		        all_indexes.add(indexes);
		    }
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		//(3) read cell_similarities
		ArrayList<Double> cell_similarities = new ArrayList<Double>(20000);
		try(BufferedReader reader = new BufferedReader(new FileReader(f_cell_similarities))){
			String line;
			    
		    while ((line = reader.readLine()) != null) {
		        double val = Double.parseDouble(line);
		        cell_similarities.add(val);
		    }
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("[Done] in "+(System.currentTimeMillis()-start)+" ms");
		return new BibleResult(k, approach, all_sims, all_indexes, cell_similarities);
	}
	
	static HashMap<Integer, BibleResult> load_seda(){
		return load("seda");
	}
	
	static HashMap<Integer, BibleResult> load_jaccard(){
		return load("jaccard");
	}
	
	static HashMap<Integer, BibleResult> load_fast_text(){
		return load("fest_text");//Note the typo
	}
	
	static BibleResult get_ground_truth() {
		return BertBibleBase.get_ground_truth();
	}
	
	/**
	 * This outputs the accuracy metric to console
	 */
	public static void compute_mapping_accuracy() {
		BibleResult bert = get_ground_truth();
		bert.out("all_indexes");
		correct_mapping(bert);
		HashMap<Integer, BibleResult> seda = load("seda");
		HashMap<Integer, BibleResult> jaccard = load("jaccard");
		HashMap<Integer, BibleResult> fast_text = load("fest_text");
		
		System.out.println("SeDA");
		for(Entry<Integer, BibleResult> e : seda.entrySet()) {
			System.out.print("k="+e.getKey()+"\t");
			correct_mapping(bert, e.getValue());
		}
		
		System.out.println("Jaccard");
		for(Entry<Integer, BibleResult> e : jaccard.entrySet()) {
			System.out.print("k="+e.getKey()+"\t");
			correct_mapping(bert, e.getValue());
		}
		
		System.out.println("FastText");
		for(Entry<Integer, BibleResult> e : fast_text.entrySet()) {
			System.out.print("k="+e.getKey()+"\t");
			correct_mapping(bert, e.getValue());
		}
	}
	
	public static void compute_similarity_correlation() {
		BibleResult bert = get_ground_truth();
		bert.out("all_indexes");
		//correct_mapping(bert);
		HashMap<Integer, BibleResult> seda = load("seda");
		HashMap<Integer, BibleResult> jaccard = load("jaccard");
		HashMap<Integer, BibleResult> fast_text = load("fest_text");
		
		System.out.println("SeDA");
		for(Entry<Integer, BibleResult> e : seda.entrySet()) {
			System.out.print("k="+e.getKey()+"\t");
			similarity_correlation(bert, e.getValue());
		}
		
		System.out.println("Jaccard");
		for(Entry<Integer, BibleResult> e : jaccard.entrySet()) {
			System.out.print("k="+e.getKey()+"\t");
			similarity_correlation(bert, e.getValue());
		}
		
		System.out.println("FastText");
		for(Entry<Integer, BibleResult> e : fast_text.entrySet()) {
			System.out.print("k="+e.getKey()+"\t");
			similarity_correlation(bert, e.getValue());
		}
	}
	
	static void similarity_correlation(BibleResult ground_truth, BibleResult to_check) {
		final int num_corpus_docs = ground_truth.all_sims[0].length;
		double[] corpus_1nn_correlation = new double[num_corpus_docs];
		
		
		for(int corpus_doc_id=0;corpus_doc_id<num_corpus_docs;corpus_doc_id++) {
			double correlation = correlation(ground_truth.all_sims, to_check.all_sims, corpus_doc_id);
			corpus_1nn_correlation[corpus_doc_id] = correlation;
		}
		
		double avg_corpus_1nn_correlation = Util.avg(corpus_1nn_correlation);
		double cell_sim_correlation = new PearsonsCorrelation().correlation(ground_truth.cell_similarities, to_check.cell_similarities);
		System.out.println("similarity_correlation()\t"+Util.outTSV(corpus_1nn_correlation)+"\tavgf\t"+avg_corpus_1nn_correlation+"\tcell\t"+cell_sim_correlation);
	}

	//FIXME Can't be fixed
	@Deprecated
	public double[][][] reconstruct_matrices(){
		final int num_docs 			   = this.all_indexes[0].length;
		final int num_lines_per_matrix = this.all_indexes.length;
		
		final int num_columns = this.cell_similarities.length / num_docs / num_lines_per_matrix;
		
		if(num_columns*num_lines_per_matrix*num_columns!=this.cell_similarities.length) {
			System.err.println("num_columns*num_lines_per_matrix*num_columns!=this.cell_similarities.length");
		}
		
		double[][][] matrices = new double[num_docs][num_lines_per_matrix][num_columns];//TODO
		int pointer = 0;
		for(int doc_id=0;doc_id<num_docs;doc_id++) {
			double[][] matrix = matrices[doc_id];
			for(int line=0;line<num_lines_per_matrix;line++) {
				for(int col=0;col<num_columns;col++){
					matrix[line][col] = this.cell_similarities[pointer]++;
				}
			}
		}
		//check if the other mebers are identic after reduce()
		return matrices;
	}
	
	private static double correlation(double[][] matrix_1, double[][] matrix_2, int corpus_doc_id) {
		// (1) get the desired feature vectors
		double[] vec_1 = new double[matrix_1.length];
		double[] vec_2 = new double[matrix_2.length];
		for(int i=0;i<vec_1.length;i++) {
			vec_1[i] = matrix_1[i][corpus_doc_id];
			vec_2[i] = matrix_2[i][corpus_doc_id];
		}
		double correlation = new PearsonsCorrelation().correlation(vec_1, vec_2);
		
		return correlation;
	}

	static final double[] thresholds = {0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0};
	public static void main(String[] args) {
		//compute_mapping_accuracy();
		compute_similarity_correlation();
		//Does not work: compute_mapping_accuracy(thresholds,"seda");
	}
	
	/*static void compute_mapping_accuracy(double[] all_thresholds, String approach) {
		BibleResult bert = get_ground_truth();
		HashMap<Integer, BibleResult> h = null;
		if(approach.toLowerCase().equals("seda")) {
			 h = load_seda();
		}else{
			System.err.println("compute_mapping_accuracy() approach="+approach);
			return;
		}
		for(double theta : all_thresholds) {
			System.out.println("SeDA theta="+theta);	
			for(Entry<Integer, BibleResult> e : h.entrySet()) {//all values of k
				System.out.print("k="+e.getKey()+"\t");
				double[][][] matrices = e.getValue().reconstruct_matrices();
				BibleResult br = new BibleResult(matrices, e.getValue());
				similarity_correlation(bert, br);
			}
		}
		
		
		
	}*/

	/**
	 * XXX Has problems with Volx Bible
	 * @return
	 */
	static int[] correct_mapping(BibleResult br) {
		int[] result = new int[br.all_indexes[0].length];
		
		for(int paragraph=0;paragraph<br.all_indexes.length;paragraph++) {
			int[] mapping_for_p = br.all_indexes[paragraph];
			for(int text=0;text<mapping_for_p.length;text++) {
				if(mapping_for_p[text]!=paragraph) {//the paragraphs of all versions should align (they don't). otherwise its an error
					result[text]++;
				}
			}
		}
		System.out.println("correct_mapping()\t"+Util.outTSV(result));

		return result;
	}
	
	/**
	 * Computes the accuracy of the mapping to_check, to a given ground truth (usually Bert)
	 * @param ground_truth
	 * @param to_check
	 * @return
	 */
	static int[] correct_mapping(BibleResult ground_truth, BibleResult to_check) {
		int[] result = new int[to_check.all_indexes[0].length];
		
		for(int paragraph=0;paragraph<to_check.all_indexes.length;paragraph++) {
			int[] mapping_for_p = to_check.all_indexes[paragraph];
			for(int text=0;text<mapping_for_p.length;text++) {
				if(mapping_for_p[text]!=paragraph) {//the paragraphs of all versions should align (they don't). otherwise its an error
					if(ground_truth.all_indexes[paragraph][text]==mapping_for_p[text]) {//or the ground truth says so
						if(text!=3) {//The volx bible is the problem
							//System.out.println("woahh");
						}
					}else{
						result[text]++;	
					}
				}else{//This one is right but the Bert ground truth is not
					if(ground_truth.all_indexes[paragraph][text]!=mapping_for_p[text]) {//or the ground truth says so
						//System.out.println("woahh");
					}
				}
			}
		}
		double sum = Util.sum(result);
		double num = to_check.all_indexes.length*to_check.all_indexes[0].length;
		System.out.println("correct_mapping()\t"+Util.outTSV(result)+"\tof\t"+num+"\t"+sum+"\t"+(1.0d-(sum/num)));

		return result;
	}

	private void out(String field) {
		if(field.equals("all_indexes")){
			for(int[] arr : this.all_indexes) {
				System.out.println(Util.outTSV(arr));
			}
		}
		
	}
}

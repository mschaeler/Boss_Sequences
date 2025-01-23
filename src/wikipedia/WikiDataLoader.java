package wikipedia;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.PorterStemmer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

import boss.hungarian.Solutions;
import boss.lexicographic.StopWords;
import boss.test.SemanticTest;
import boss.util.Config;

public class WikiDataLoader {
	public static final int THOUSAND = 1000;
	//static final int[] intput_sequence_length = {2*THOUSAND, 20*THOUSAND, 200*THOUSAND, 2*THOUSAND*THOUSAND};
	static final int[] intput_sequence_length = {2*THOUSAND, 4*THOUSAND, 8*THOUSAND, 16*THOUSAND, 2*16*THOUSAND, 3*16*THOUSAND, 4*16*THOUSAND, 5*16*THOUSAND};//, 5*16*THOUSAND, 6*16*THOUSAND, 7*16*THOUSAND, 8*16*THOUSAND, 9*16*THOUSAND};
	//static final int[] intput_sequence_length = {16*THOUSAND};
	//int[] all_solutions = {SemanticTest.NAIVE, SemanticTest.BASELINE, SemanticTest.SOLUTION};
	int[] all_solutions = {6};
	
	static String folder = "./data/wikipedia/";
	//static String test_file = "test-5000.txt";
	static String test_file = "wiki-1024000.txt";
	static String embedding_path = "all_words_wiki.tsv";
	
	static boolean header_written = false;
	static String result_path = "./results/wiki_results"+System.currentTimeMillis()+".tsv";
	
	public static final int print_nothing 		= -1;
	public static final int print_final_token 	= 0;
	public static final int print_everything 	= 1;
	static double threshold = 0.7;
	int num_repititions = 1;
	/**
	 * -1 print nothing
	 * 0 print final tokens
	 * 1 print everything
	 */
	public static int verbose_level = 1;
	
	ArrayList<int[]> raw_paragraphs_b1;
	ArrayList<int[]> raw_paragraphs_b2;
	HashMap<Integer, double[]> embedding_vector_index;
	
	String load_file(String file_name) {
		File f = new File(folder+file_name);
		if(!f.exists()) {
			System.err.println("!f.exists()");
			return null;
		}
		StringBuilder resultStringBuilder = new StringBuilder();
	    try (FileReader in = new FileReader(f); BufferedReader br = new BufferedReader(in);) {
	        String line;
	        while ((line = br.readLine()) != null) {
	            resultStringBuilder.append(line).append("\n");
	        }
	    } catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return resultStringBuilder.toString();
	}
	
	void prepare_solution(ArrayList<String> tokens) {
		System.out.println("Words after pre-processing= "+tokens.size());
		HashSet<String> unique_tokens = new HashSet<String>(tokens.size());
		for(String s : tokens) {
			unique_tokens.add(s);
			//System.out.println(s);
		}
		System.out.println("Unique words after pre-processing= "+unique_tokens.size());
		ArrayList<String> unique_tokens_sorted = new ArrayList<String>(unique_tokens.size());
		for(String s : unique_tokens) {
			unique_tokens_sorted.add(s);
		}
		System.out.println("unique_tokens_sorted");
		Collections.sort(unique_tokens_sorted);
		/*for(String s : unique_tokens_sorted) {
			System.out.println(s);
		}*/
		
		HashMap<String, Integer> token_ids = SemanticTest.strings_to_int(unique_tokens_sorted);
		this.embedding_vector_index = SemanticTest.create_embedding_vector_index(token_ids, unique_tokens_sorted, folder+embedding_path);
		
		int mid = tokens.size() / 2;
		ArrayList<String> book_1 = new ArrayList<String>(mid);
		ArrayList<String> book_2 = new ArrayList<String>(mid);
		for(int i=0;i<mid;i++) {
			book_1.add(tokens.get(i));
			book_2.add(tokens.get(mid+i));
		}
		
		this.raw_paragraphs_b1  = encode(book_1, token_ids);
		this.raw_paragraphs_b2  = encode(book_2, token_ids);
		
		//out(book_1, raw_paragraphs_b1.get(0));
		//out(book_2, raw_paragraphs_b2.get(0));
	}
	
	void out(ArrayList<String> raw_book, int[] raw_paragraphs) {
		if(raw_book.size()!=raw_paragraphs.length) {
			System.err.println("out(String[] raw_book, int[] raw_paragraphs) raw_book.length!=raw_paragraphs.length");
		}
		String words = "";
		String ids = "";
		for(int i=0;i<raw_book.size();i++) {
			words += raw_book.get(i)+"\t";
			ids += raw_paragraphs[i]+"\t";
		}
		System.out.println(words);
		System.out.println(ids);
		
	}
	
	ArrayList<int[]> encode(ArrayList<String> raw_book, HashMap<String, Integer> token_ids) {
		ArrayList<int[]> result = new ArrayList<int[]>(1);
				
		int[] paragraph_token_ids = new int[raw_book.size()];
		for(int i=0;i<paragraph_token_ids.length;i++) {
			String token = raw_book.get(i);
			Integer id = token_ids.get(token);
			if(id!=null) {
				paragraph_token_ids[i] = id.intValue();
			}else{
				System.err.println("id==null for "+token);
			}
		}
		result.add(paragraph_token_ids);
		return result;
	}
	
	public static void run(){
		new WikiDataLoader().run(test_file);
	}
	
	private void run(String file) {
		System.out.println("WikiDataLoader.run()");
		String line = load_file(file);
		ArrayList<String> tokens = tokenize_txt_align(line);
		//out_unique_tokens(tokens);
		System.out.println("tokenize() returned "+tokens.size()+ " tokens");
		for(int length : intput_sequence_length) {
			System.out.println("*** Length = "+length);
			if(length<=tokens.size()) {
				ArrayList<String> input = shorten_to_length(tokens, length);
				System.out.println(input);
				prepare_solution(input);
				for(int solution_enum : all_solutions) {
					run_solution(solution_enum);	
				}
				System.gc();
			}else{
				System.err.println("length>line.length()");
			}
		}
	}

	private void out_unique_tokens(ArrayList<String> tokens) {
		HashSet<String> unique_tokens = new HashSet<String>(tokens.size());
		for(String s : tokens) {
			unique_tokens.add(s);
			//System.out.println(s);
		}
		System.out.println("Unique words after pre-processing= "+unique_tokens.size());
		ArrayList<String> unique_tokens_sorted = new ArrayList<String>(unique_tokens.size());
		for(String s : unique_tokens) {
			unique_tokens_sorted.add(s);
		}
		System.out.println("unique_tokens_sorted");
		Collections.sort(unique_tokens_sorted);
		for(String s : unique_tokens_sorted) {
			System.out.println(s);
		}
	}

	private ArrayList<String> shorten_to_length(ArrayList<String> tokens, int length) {
		ArrayList<String> ret = new ArrayList<String>(length);
		for(int i=0;i<length;i++){
			ret.add(tokens.get(i));
		}
		return ret;
	}

	private void run_solution(int solution_enum) {
		ArrayList<double[]> all_run_times = new ArrayList<double[]>();
		double[] run_times=null;
		
		for(int k : Config.wiki_k_s) {
			Solutions.dense_global_matrix_buffer = null;
			Solutions s = new Solutions(raw_paragraphs_b1, raw_paragraphs_b2, k, threshold, embedding_vector_index);
			int repitions = 0;
			double run_time = 0;
			while(repitions++<num_repititions) {
				if(solution_enum == SemanticTest.SOLUTION) {
					run_times = s.run_solution();
				}else if(solution_enum == SemanticTest.BASELINE) {
					run_times = s.run_baseline();
				}else if(solution_enum == SemanticTest.NAIVE) {
					run_times = s.run_naive();
				}else if(solution_enum == 6) {//TODO FastText
					run_times = s.run_fast_text();
				}else{
					System.err.println("SemanticTest.run() unknown solution enum: "+solution_enum);
				}
				run_time += run_times[0];
			}
			run_time /= repitions-1;
			double[] temp = {run_time};
			all_run_times.add(temp);
		}
		for(int i=0;i<Config.wiki_k_s.length;i++) {
			System.out.print("k="+Config.wiki_k_s[i]+"\t");
		}
		System.out.println();
		
		for(int p=0;p<all_run_times.get(0).length;p++) {
			for(int i=0;i<Config.wiki_k_s.length;i++) {
				run_times = all_run_times.get(i);
				System.out.print(run_times[p]+"\t");
			}
			System.out.println();
		}
		boolean RESULTS_TO_FILE = true;
		if(RESULTS_TO_FILE) {
			//String result_path = "./results/pan_results_"+System.currentTimeMillis()+".tsv";
		    try {
		        BufferedWriter output = new BufferedWriter(new FileWriter(result_path, true));

		        // Writes the string to the file
		        if(!header_written) {
			        for(int i=0;i<Config.wiki_k_s.length;i++) {
			        	output.write("k="+Config.wiki_k_s[i]+"\t");
					}
					output.newLine();
					header_written = true;
		        }
				
				for(int p=0;p<all_run_times.get(0).length;p++) {
					for(int i=0;i<Config.wiki_k_s.length;i++) {
						run_times = all_run_times.get(i);
						output.write(run_times[p]+"\t");
					}
					output.newLine();
				}
		        
		        // Closes the writer
		        output.close();
		      }catch (Exception e) {
		          e.getStackTrace();
		      }
		}
		all_run_times.clear();
	}
	
	ArrayList<String> tokenize_txt_align(String org){
		HashSet<String> stopwords = StopWords.get_DONG_DENG_STOPWORDS();		
		Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_35, stopwords);
		
		if(verbose_level>=print_everything) {System.out.println("org=\t\t\t"+org.subSequence(0, Math.min(1000, org.length()-1)));}
		final String delim = "[\"#$%&\'()*+,-./:;<=>?@\\[\\]^_`{|}~ ]";
		// final String period_str = "\n\t\r\x0b\x0c,.!?;!";
		final String period_str = "[\n\t\r,.!?;!]";// no line tabs in Java
		
		String[] sentences = org.split(period_str);
		if(verbose_level>=print_everything) {System.out.println("sentences=\t\t"+Arrays.toString(sentences));}
		
		ArrayList<String> ret = new ArrayList<String>();
		PorterStemmer stemmer = new PorterStemmer();
		
		for(String sentence : sentences) {
			String[] tokens = sentence.split(delim);
			for(String t : tokens) {
				Reader r = new StringReader(t);
				TokenStream ts = analyzer.tokenStream("field", r);
				try {
					ts.reset();
					while (ts.incrementToken()) {
					    t = ts.getAttribute(CharTermAttribute.class).toString();
					    if(!stopwords.contains(t)) {//@FIX in original code
							String stem_word = stemmer.stem(t.toLowerCase());
							if(!stopwords.contains(stem_word)) {
								ret.add(stem_word);	
							}
					    }
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		analyzer.close();
		return ret;
	}
}

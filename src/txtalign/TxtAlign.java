package txtalign;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.lucene.analysis.PorterStemmer;

import boss.load.Importer;
import boss.test.SemanticTest;
import boss.util.Config;
import pan.MatrixLoader;

public class TxtAlign {
	private static final int MY_NAN = -999;

	static int MAXLENGTH = 0x7fffffff;

	static String output_folder = "output";
	static String query_path = Importer.PAN11_PREFIX_SUSP.replace("suspicious-document", "");

	// int K = 128;
	// double theta = 0.8;
	static double theta = 0.3;
	static int K = 16;
	static int tau = 16;

	static String src_path = Importer.PAN11_PREFIX_SRC.replace("source-document", ""); // path of the source folder
	// threshold
	static HashMap<String, Integer> word2id; // map from word to id
	static ArrayList<String> id2word; // map from id to word
	static ArrayList<Integer> orginal_pos; // stores the original place for words

	static int hash_func = 0;
	static ArrayList<Pair> ab = new ArrayList<Pair>();

	public static void main(String[] args) {
		System.out.println("This is txtalign-index-sentence-search");
		double start = System.currentTimeMillis();

		for (int i = 0; i < args.length; i++) {
			String arg = args[i];
			if (arg == "-src_path")
				src_path = args[i + 1];
			if (arg == "-query_path")
				query_path = args[i + 1];
			if (arg == "-theta")
				theta = Double.parseDouble(args[i + 1]);
			if (arg == "-k")
				K = Integer.parseInt(args[i + 1]);
			if (arg == "-doc_max_len")
				MAXLENGTH = Integer.parseInt(args[i + 1]);
			if (arg == "-tau")
				tau = Integer.parseInt(args[i + 1]);
		}
		System.out.println("current k is: " + K);
		System.out.println("current theta is: " + theta);
		System.out.println("src_path is: " + src_path);
		System.out.println("susp_path is: " + query_path);
		// 0. generate hash functions
		generateHashFunc(1111, ab);
		// 1. read stopwords from stopwords.txt
		HashSet<String> stopWords = new HashSet<String>();
		for (String sw : boss.lexicographic.StopWords.DONG_DENG_STOPWORDS) {// store stopwords
			stopWords.add(sw);
		};
		// buildDic(src_path, query_path, stopWords, "all_words.txt");

		HashMap<String, Integer> word2id = read_all_words("./data/all_words.txt");
		// TODO make cofigurable loadWords("all_words.txt", word2id);
		//word2id = SemanticTest.w2i_pan();

		HashMap<String, Integer> word2fre = new HashMap<String, Integer>();
		// loadWordsFre("frequency.txt", word2fre);
		System.out.println("the size for word2id is: " + word2id.size());

		// cout << "the size for word2fre is: " << word2fre.size() << endl;
		/*
		 * float sim =
		 * getWeightedSim("01-manual-obfuscation/src/source-document05889.txt", 581,
		 * 490, "01-manual-obfuscation/susp/suspicious-document00228.txt", 1925, 520,
		 * word2fre); cout << "weighted jac is: " << sim << endl; return 0;
		 */
		double stop = System.currentTimeMillis();
		double duration = (stop - start);
		System.out.println("preprocissing time 1: " + duration + " ms");
		start = System.currentTimeMillis();

		// 2. read words from source folder
		ArrayList<String> files = new ArrayList<String>(); // store file_path of each document in the given folder
		getFiles(src_path, files);

		int doc_num = files.size();
		ArrayList<ArrayList<Integer>> docs = new ArrayList<ArrayList<Integer>>(doc_num);
		ArrayList<ArrayList<Integer>> docs_ppos = new ArrayList<ArrayList<Integer>>(doc_num);
		ArrayList<ArrayList<Integer>> docs_offset = new ArrayList<ArrayList<Integer>>(doc_num);
		for(int id=0;id<doc_num;id++) {
			docs.add(new ArrayList<Integer>());
			docs_ppos.add(new ArrayList<Integer>());
			docs_offset.add(new ArrayList<Integer>());
		}
		int did = 0;
		int total_doc_size = 0;
		for (String file : files) {
			word2int(file, docs.get(did), docs_ppos.get(did), docs_offset.get(did), word2id, id2word, stopWords);
			total_doc_size += docs.get(did).size();
			did++;
		}

		stop = System.currentTimeMillis();
		duration = (stop - start);
		System.out.println("readfile time: " + duration + " ms");
		System.out.println("doc number: " + doc_num);
		start = System.currentTimeMillis();

		long total_window_num = 0;
		// generate composite windows for source folder
		// store compact windows of each document in the given folder
		ArrayList<ArrayList<CompositeWindow>> docs_cws = new ArrayList<ArrayList<CompositeWindow>>(doc_num); 
		for(int i=0;i<doc_num;i++) {
			docs_cws.add(new ArrayList<CompositeWindow>());
		}
		for (int docid = 0; docid < doc_num; docid++) {
			// Generate2KSketch(app_docs_cws[docid], docs[docid], docs_ppos[docid]);
			// brute_force(docs[docid]);//TODO
			SketchGeneration(docs_cws.get(docid), docs.get(docid), docs_ppos.get(docid), ab);
		}

		stop = System.currentTimeMillis();
		duration = (stop - start);
		System.out.println("average window num: " + ((double) total_window_num / (double) doc_num));
		System.out.println("source window generation time: " + duration + " ms");

		ArrayList<String> query_files = new ArrayList<String>();
		getFiles(query_path, query_files);
		System.out.println("query files size: " + query_files.size());
		// for the filtering part

		start = System.currentTimeMillis();
		HashMap<Integer, ArrayList<Pair>> inverted_index = new HashMap<Integer, ArrayList<Pair>>();
		ArrayList<Pair> cw_ids = new ArrayList<Pair>();
		buildIndex(docs_cws, inverted_index, cw_ids);
		stop = System.currentTimeMillis();
		duration = (stop - start);
		System.out.println("build inverted index time is: " + duration + " ms");

		start = System.currentTimeMillis();

		ArrayList<Pair> cnt = new ArrayList<Pair>(cw_ids.size());
		int timestamp = 2;
		long res_size = 0;

		for (String query_file : query_files) {
			ArrayList<Integer> query = new ArrayList<Integer>();
			ArrayList<Integer> query_ppos = new ArrayList<Integer>();
			ArrayList<Integer> query_offset = new ArrayList<Integer>();
			ArrayList<CompositeWindow> query_cw = new ArrayList<CompositeWindow>();
			word2int(query_file, query, query_ppos, query_offset, word2id, id2word, stopWords);
			SketchGeneration(query_cw, query, query_ppos, ab);
			System.out.println("query sketch num " + query_cw.size());
			ArrayList<ArrayList<windowPair>> vec_pairs = new ArrayList<ArrayList<windowPair>>(doc_num);
			Search(inverted_index, cw_ids, cnt, docs_cws, query_cw, docs_offset, query_offset, vec_pairs, timestamp);
			// res_size += vec_pairs.size();
			System.out.println("result num " + vec_pairs.size());
			System.out.println("current deal with " + query_file);
			ArrayList<ArrayList<windowPair>> ret_pairs = new ArrayList<ArrayList<windowPair>>();
			ArrayList<String> valid_files = new ArrayList<>();
			int docid = 0;
			System.out.println(
					"----------------------------------------------------------------------------------------");
			for (ArrayList<windowPair> ori_pair : vec_pairs) {
				if (ori_pair.size() == 0) {
					docid += 1;
					continue;
				}
				ArrayList<windowPair> reduced = new ArrayList<windowPair>();
				System.out.println("possible candidate file: " + files.get(docid));
				// reduceOperation(ori_pair, reduced);
				String dName = files.get(docid).substring(files.get(docid).length() - 24, files.get(docid).length());
				reduceByRealJac(files.get(docid), query_file, ori_pair, reduced, word2id, stopWords, word2fre);
				if (reduced.size() != 0) {
					valid_files.add(dName);
					ret_pairs.add(reduced);
				}

				docid += 1;
			}

			System.out.println(
					"----------------------------------------------------------------------------------------");
			String qName = query_file.substring(query_file.length() - 28, query_file.length());
			if (ret_pairs.size() != 0) {
				System.out.println("generate XML now");
				generateXML(output_folder, qName, valid_files, ret_pairs);
			}
		}

		stop = System.currentTimeMillis();
		duration = (stop - start);
		System.out.println("total query time is: " + duration + " ms");
		System.out.println("average query time is: " + (duration / (double) query_files.size()) + " ms");
		System.out.println("average result size is " + (res_size * 1.0 / (double) query_files.size()));
	}

	private static void buildIndex(ArrayList<ArrayList<CompositeWindow>> docs_cw,
			HashMap<Integer, ArrayList<Pair>> inverted_index, ArrayList<Pair> cw_ids) {
	    for (int i = 0; i < docs_cw.size(); i++) 
	    {
	        for (int j = 0; j < docs_cw.get(i).size(); j++) 
	        {
	            int current_id = cw_ids.size();
	            cw_ids.add(new Pair(i, j));
	            //sort(docs_cw[i][j].sketch.begin(), docs_cw[i][j].sketch.end());//XXX why not sorting the corresponding positions as well?
	            Collections.sort(docs_cw.get(i).get(j).sketch);
	            for (int x = 0; x < K; x++) 
	            {
	                int hv = docs_cw.get(i).get(j).sketch.get(x);
	                inverted_index.get(hv).add(new Pair(current_id, x));
	            }
	        }
	    }

	}

	private static void generateXML(String folder_name, String qName, ArrayList<String> doc_names,
			ArrayList<ArrayList<windowPair>> pairs) {
	    //String fn = folder_name + "/" + qName.substring(0, qName.length() - 4) + ".xml";
	    
	    System.out.println("<?xml version=\"1.0\" encoding=\"utf-8\"?><document reference=\"" + qName + "\">");
	    for (int i = 0; i < pairs.size(); i++){
	        String dName = doc_names.get(i);
	        for (int j = 0; j < pairs.get(i).size(); j++){
	            System.out.print("<feature name=\"detected-plagiarism\" this_offset=\"" + pairs.get(i).get(j).queryOffset);
	            System.out.print("\" this_length=\"" + pairs.get(i).get(j).queryLen + "\" source_reference=\"" + dName + "\" source_offset=\"");
	            System.out.println(pairs.get(i).get(j).docOffset + "\" source_length=\"" + pairs.get(i).get(j).docLen + "\"/>");
	        }
	    }
	    System.out.println("</document>");
	}

	private static void reduceByRealJac(String src_file, String susp_file, ArrayList<windowPair> pairs,
			ArrayList<windowPair> reduced, HashMap<String, Integer> word2id2, HashSet<String> stopWords,
			HashMap<String, Integer> word2fre) {
		//get the jaccard from each window pair
	    double jac = 0.0;
	    double weight_sim = 0.0;
	    int index = 0;
	    //int weight_index = 0;
	    
	    System.out.println("before reduce operation, there are " + pairs.size() + " possible candidates");
	    System.out.println("current susp file is: " + susp_file + " current src file is: " + src_file);
	    
	    int index_record = 0;
	    for (int i = 0; i < pairs.size() && i < 2000; i++){
	        int src_offset = pairs.get(i).docOffset;
	        int src_length = pairs.get(i).docLen;
	        int susp_offset = pairs.get(i).queryOffset;
	        int susp_length = pairs.get(i).queryLen;
	        index_record += 1;
	        if ((src_length > 1.3 * susp_length || susp_length > 1.3 * src_length || susp_length >= 3000 || src_length >= 3000)){
	            System.out.println("it works for length drop");
	            continue;
	        }
	        double temp_jac = getRealJaccard(src_file, src_offset, src_length, susp_file, susp_offset, susp_length, word2id, stopWords);
	        //float temp_weight_sim = getWeightedSim(src_file, src_offset, src_length, susp_file, susp_offset, susp_length, word2fre);
	        
	        if (temp_jac > jac){
	            jac = temp_jac;
	            index = i;
	        }
	        /*
	        if (temp_weight_sim > weight_sim){
	            weight_sim = temp_weight_sim;
	            weight_index = i;
	        }
	        */
	    }
	    if (jac  >= theta ){
	        reduced.add(pairs.get(index));
	    }else{
	        if (index_record == 2000){
	            System.out.println("need more operations ");
	            while (index_record < 20000 && index_record < pairs.size()){
	                int src_offset = pairs.get(index_record).docOffset;
	                int src_length = pairs.get(index_record).docLen;
	                int susp_offset = pairs.get(index_record).queryOffset;
	                int susp_length = pairs.get(index_record).queryLen;
	                double temp_jac = getRealJaccard(src_file, src_offset, src_length, susp_file, susp_offset, susp_length, word2id, stopWords);
	                // float temp_weight_sim = getWeightedSim(src_file, src_offset, src_length, susp_file, susp_offset, susp_length, word2fre);
	                index_record += 1;
	                
	                if (temp_jac >= theta){
	                    System.out.println("more operations work");
	                    reduced.add(pairs.get(index_record));
	                    break;
	                }
	            }
	        }
	    }
	    System.out.println("the largest weight sim is: " + weight_sim + ", the largest jaccard is: " + jac);
	    System.out.println("after reduce operation, " + reduced.size() + " pairs left");

	}

	//get read jaccard
	static double getRealJaccard(final String src_name, int src_offset, int src_length, final String susp_name, int susp_offset, int susp_length, HashMap<String, Integer> word2id, HashSet<String> stopWords){
	    //read from source file
	    /*ifstream src_file(src_name, ios::binary);
	    src_file.seekg(src_offset + 3);
	    String src_content(src_length + 1, '\0');
	    src_file.read(&src_content[0], src_length);
*/
	    String all_src_content = read(src_name);//XXX why again, for timing experiments read only relevant part, why this +3?
	    String src_content = all_src_content.substring(src_offset+3, src_length);
	    
	    //read from susp file
	    /*ifstream susp_file(susp_name, ios::binary);
	    susp_file.seekg(susp_offset + 3);
	    string susp_content(susp_length + 1, '\0');
	    susp_file.read(&susp_content[0], susp_length);
	    */
	    String all_susp_content = read(susp_name);
	    String susp_content = all_susp_content.substring(susp_offset+3, susp_length);
	    
	    //split content into words
	    ArrayList<Integer> src_words = new ArrayList<Integer>();
	    ArrayList<Integer> susp_words= new ArrayList<Integer>();
	    /*
	    //debug
	    cout << "src_content: " << src_content << endl;
	    cout << "-----------------------------------------------------------------" << endl;
	    cout << "susp content: " << susp_content << endl;
	    */
	    final String delim = "\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\r ,.!?;!";
	    ArrayList<String> tokens = new ArrayList<String>();
	    ArrayList<Integer> tokensOffsets = new ArrayList<Integer>();
	    // read it word by word from src_content
	    strToTokens(src_content, delim, tokens, tokensOffsets);
	    for (int i = 0; i < tokens.size(); i++){
	        String stem_word = stem(tokens.get(i));
	        if (stopWords.contains(stem_word))
	            continue;
	        Integer wid = word2id.get(stem_word);
	        if(wid==null) {
	        	System.err.println(wid==null);
	        }
	        src_words.add(wid); 
	    }
	    tokens.clear();
	    //cout << "src words size: " << src_words.size() << endl;
	    // read it word by word from src_content
	    strToTokens(susp_content, delim, tokens, tokensOffsets);
	    for (int i = 0; i < tokens.size(); i++){
	    	String stem_word = stem(tokens.get(i));
	        if (stopWords.contains(stem_word))
	            continue;
	        Integer wid = word2id.get(stem_word);
	        if(wid==null) {
	        	System.err.println(wid==null);
	        }
	        susp_words.add(wid); 
	    }
	    //cout << "susp words size: " << susp_words.size() << endl;
	    ArrayList<Integer> inter_ret = new ArrayList<Integer>();
	    ArrayList<Integer> union_ret = new ArrayList<Integer>();
	    getIntersection(susp_words, src_words, inter_ret);
	    getUnion(susp_words, src_words, union_ret);
	    double real_jac = ((double)inter_ret.size()) / (double)union_ret.size();
	    return real_jac;
	    /*
	    cout << "real_jac is: " << real_jac << endl;
	    if (real_jac >= theta){
	        return true;
	    }else{
	        return false;
	    }
	    */
	}
	
	private static void getUnion(ArrayList<Integer> susp_words, ArrayList<Integer> src_words,
			ArrayList<Integer> ret) {
	    HashSet<Integer> s = new HashSet<Integer>();
	    for (Integer item : susp_words){
	        s.add(item);
	    }
	    for (Integer item : src_words){
	        s.add(item);
	    }
	   
	    for (Integer item : s){
	       ret.add(item);
	    }
	}

	private static void getIntersection(ArrayList<Integer> susp_words, ArrayList<Integer> src_words,
			ArrayList<Integer> ret) {
		HashSet<Integer> s1 = new HashSet<Integer>();
	    for (Integer item : susp_words){
	        s1.add(item);
	    }
	    for (Integer item: src_words){
	        if (!s1.contains(item)){
	            ret.add(item);
	        }
	    }
	    Collections.sort(ret);//XXX
	}
	
	static PorterStemmer stemmer = new PorterStemmer();
	private static String stem(String string) {
		return stemmer.stem(string);
	}

	private static void Search(HashMap<Integer, ArrayList<Pair>> inverted_index, ArrayList<Pair> cw_ids,
			ArrayList<Pair> cnt, ArrayList<ArrayList<CompositeWindow>> docs_cw, ArrayList<CompositeWindow> query_cw,
			ArrayList<ArrayList<Integer>> docs_pos, ArrayList<Integer> query_pos,
			ArrayList<ArrayList<windowPair>> vec_pairs, int timestamp) {
	    int num_hit = (int) Math.ceil( K * theta );
	    //cout << "num_hit: " << num_hit << endl;
	    int wid = 0;
	    for (CompositeWindow cw: query_cw)
	    {
	        // cout << "processing" << wid++ << endl;
	        timestamp += 1;
	        Collections.sort(cw.sketch);
	        for (int i = 0; i < K; i++)
	        {
	            int hv = cw.sketch.get(i);
	            if (inverted_index.containsKey(hv))
	            {
	                // cout << "inverte list length: " << inverted_index[hv].size() << endl;
	                for (Pair id_pos: inverted_index.get(hv))
	                {
	                    int id = id_pos.first;
	                    int pos = id_pos.second;
	                    if (cnt.get(id).first != timestamp)
	                    {
	                    	cnt.get(id).first = timestamp;
	                    	cnt.get(id).second = 0;
	                    }
	                    if (i + pos - cnt.get(id).second < K)
	                    {
	                    	cnt.get(id).second += 1;
	                        if (cnt.get(id).second == num_hit)
	                        {
	                            int docid = cw_ids.get(id).first;
	                            int cwid = cw_ids.get(id).second;
	                            int doc_left = docs_cw.get(docid).get(cwid).beg_range.l;
	                            int doc_right = docs_cw.get(docid).get(cwid).end_range.r;
	                            int doc_offset = docs_pos.get(docid).get(doc_left);
	                            int doc_length = docs_pos.get(docid).get(doc_right) - doc_offset + 1;
	                            int query_left = cw.beg_range.l;
	                            int query_right = cw.end_range.r;
	                            int query_offset = query_pos.get(query_left);
	                            int query_length = query_pos.get(query_right) - query_offset + 1;
	                            vec_pairs.get(docid).add(new windowPair(doc_offset, doc_length, query_offset, query_length));
	                        }
	                    }    
	                }
	            }
	        }
	    }

	}

	static int other_id = -2;
	private static void word2int(String file, ArrayList<Integer> doc, ArrayList<Integer> ppos,
			ArrayList<Integer> offsets, HashMap<String, Integer> word2id, ArrayList<String> id2word,
			HashSet<String> stopwords) {
		// Static variables across the function
		//final int mulWordId = -1;

		final String delim = "[\"#$%&\'()*+,-./:;<=>?@\\[\\]^_`{|}~ ]";
		// final String period_str = "\n\t\r\x0b\x0c,.!?;!";
		final String period_str = "[\n\t\r,.!?;!]";// no line tabs in Java

		// Read from file ...
		String docstr = read(file);

		// Make the docstr to sentences ..
		ArrayList<String> sentences = new ArrayList<String>();
		ArrayList<Integer> sentenceOffsets = new ArrayList<Integer>();
		strToTokens(docstr, period_str, sentences, sentenceOffsets);

		ppos.add(0);
		for (int s = 0; s < sentences.size(); s++) {
			// cout << "sentence: " << sentences[s] << endl;
			// get words in this sentence
			ArrayList<String> tokens = new ArrayList<String>();
			ArrayList<Integer> wordOffsets = new ArrayList<Integer>();
			strToTokens(sentences.get(s), delim, tokens, wordOffsets);

			int offset = sentenceOffsets.get(s);
			// Iterate each word in the sentence
			for (int i = 0; i < tokens.size(); i++) {
				String stem_word = stem(tokens.get(i));
				// Skip stop words
				if (stopwords.contains(stem_word))// XXX this is the problematic line
					continue;
				/*
				 * //modify here because word2id has already been generated // If a new word,
				 * add to word2id if (word2id.find(stem_word) == word2id.end()) {
				 * word2id[stem_word] = id2word.size(); id2word.emplace_back(stem_word); }
				 */

				Integer wid = word2id.get(stem_word);
				if(Config.USE_TXT_ALIGN_FIX) {
					if(wid==null) {
						wid = word2id.get(stem_word.toLowerCase());
					}
				}
				if (wid == null) {
					wid = other_id;
					if(Config.USE_TXT_ALIGN_FIX) {
						word2id.put(stem_word, wid);
					}
					System.err.println("wid == null for \""+stem_word +"\" using "+other_id-- +" in "+file+" Sentence "+s);
				}
				doc.add(wid);
				offsets.add(wordOffsets.get(i) + offset);
			}

			// Record the period position
			if (ppos.size() != doc.size()) {
				ppos.add(doc.size());// Does this work?
			}
		}
	}

	public static HashMap<String, Integer> read_all_words(String file) {
		HashMap<String, Integer> w2i = new HashMap<String, Integer>(400000);//about 400k words in the file 
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {

			String line = br.readLine();
			int id = 0;
			while (line != null) {
				w2i.put(line, id++);
				line = br.readLine();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return w2i;
	}
	
	private static String read(String file) {
		StringBuilder sb = new StringBuilder();
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {

			String line = br.readLine();

			while (line != null) {
				sb.append(line);
				sb.append(System.lineSeparator());
				line = br.readLine();
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return sb.toString();
	}

	static void strToTokens(String str, final String delimiter, ArrayList<String> res, ArrayList<Integer> offsets) {
		/*
		 * // Replace illegal chars for (int i = 0; i < str.length(); ++i) { str[i] =
		 * str[i] <= 0 || str[i] == '\n'? ' ': str[i]; }
		 */

		char[] inputStr = str.toCharArray();
		String[] key = str.split(delimiter);

		int offset = 0;
		for (String t : key) {
			int startPos = offset;// TODO and check
			{
				char[] temp = t.toCharArray();
				for (int i = 0; i < temp.length; i++) {
					if (inputStr[offset + i] != temp[i]) {
						System.err.println("inputStr[offset+i]!=temp[i]");
					}
				}
			}
			offsets.add(startPos);
			res.add(t.toLowerCase());
			offset += t.length() + 1;
		}
	}

	private static void getFiles(String dir, ArrayList<String> files) {
		File directory = new File(dir);
		if(!directory.exists()) {
			System.err.println(dir+" does not exist");
		}
		List<String> temp = null;
	    try (Stream<Path> stream = Files.list(Paths.get(dir))) {
	        temp=stream
	          .filter(file -> !Files.isDirectory(file))
	          .map(Path::getFileName)
	          .map(Path::toString)
	          .collect(Collectors.toList());
	    }catch (Exception e) {
			System.err.println(e);
		}
		
		for (String s : temp) {
			files.add(dir+s);
		}
	}

	private static void loadWords(String string, HashMap<String, Integer> word2id2) {
		// TODO Auto-generated method stub

	}

	// The hash value function
	static final int hval(final ArrayList<Pair> hf, int word) {
		return hval(hf, word, 0);
	}

	static final int hval(final ArrayList<Pair> hf, int word, final int kth_hash) {
		return hf.get(kth_hash).first * word + hf.get(kth_hash).second;
	}

	static void SketchGeneration(ArrayList<CompositeWindow> res_cws, final ArrayList<Integer> doc,
			final ArrayList<Integer> ppos, final ArrayList<Pair> hf) {
		ArrayList<Pair> words_hash_pos = new ArrayList<Pair>(doc.size());
		for (int word_pos = 0; word_pos < doc.size(); word_pos++) {
			words_hash_pos.add(new Pair(hval(hf, doc.get(word_pos)), word_pos));
		}

		// first sort by hash value and second by position
		Collections.sort(words_hash_pos);

		// build the occurrence table, one double linked list
		Node[] occurrences = new Node[doc.size()];
		for (int i = 0; i < occurrences.length; i++) {
			occurrences[i] = new Node();
		}
		int prev_hv = words_hash_pos.get(0).first + 1;
		int prev_pos = -1;
		// for (auto it = words_hash_pos.begin(); it != words_hash_pos.end(); ++it)
		for (Pair it : words_hash_pos) {
			int hv = it.first;
			int pos = it.second;
			// cout << "pos: " << pos << " hash " << hv << endl;
			if (hv == prev_hv) {
				occurrences[pos].prev = prev_pos;
				occurrences[prev_pos].next = pos;
			} else {
				if (prev_pos != -1)
					occurrences[prev_pos].next = Integer.MAX_VALUE;

				prev_hv = hv;
				occurrences[pos].prev = Integer.MIN_VALUE;
			}
			prev_pos = pos;
		}
		occurrences[prev_pos].next = Integer.MAX_VALUE;

		// maintain the skip list using double linked list trick
		int doc_size = doc.size();
		ArrayList<Node> linkedlist = new ArrayList<Node>();

		Node HEAD = new Node(MY_NAN, 0);
		Node TAIL = new Node(doc_size - 1, MY_NAN);
		int HEAD_ID = -1;
		int TAIL_ID = doc_size;

		for (int word_pos = 0; word_pos < doc.size(); word_pos++) {
			linkedlist.add(new Node(word_pos - 1, word_pos + 1));
		}

		// keep the neighbors when visiting
		Node[] neighbors = new Node[doc.size()];
		for (int i = 0; i < neighbors.length; i++) {
			neighbors[i] = new Node();
		}
		// reverse iteration
		ListIterator<Pair> li = words_hash_pos.listIterator(words_hash_pos.size());
		while (li.hasPrevious()) {
			Pair rit = li.previous();
			int pos = rit.second;
			// first record the two neighbors of pos in current linked list
			neighbors[pos].next = linkedlist.get(pos).next;
			neighbors[pos].prev = linkedlist.get(pos).prev;

			// next update the linked list by removing pos
			int next = linkedlist.get(pos).next;
			int prev = linkedlist.get(pos).prev;

			if (next != TAIL_ID)
				linkedlist.get(pos).prev = prev;
			else
				TAIL.prev = prev;

			if (prev != HEAD_ID)
				linkedlist.get(pos).next = next;
			else
				HEAD.next = next;
		}
		// initialize a linked list for D;
		Node[] D = new Node[doc.size() + 1];
		for (int i = 0; i < D.length; i++) {
			D[i] = new Node();
		}

		// cout << "visiting words" << endl;
		// now, visiting in ascending order
		Node[] skiplist = new Node[doc.size()];
		for (int i = 0; i < skiplist.length; i++) {
			skiplist[i] = new Node();
		}
		Node SKIP_HEAD = new Node(MY_NAN, TAIL_ID);
		Node SKIP_TAIL = new Node(HEAD_ID, MY_NAN);
		for (Pair it : words_hash_pos) {
			// cout << "pos: " << it->second << " hash " << it->first << endl;
			int pos = it.second;
			// first insert into the skip list
			int next = neighbors[pos].next;
			int prev = neighbors[pos].prev;

			skiplist[pos].next = next;
			skiplist[pos].prev = prev;

			if (next != TAIL_ID)
				skiplist[next].prev = pos;
			else
				SKIP_TAIL.prev = pos;

			if (prev != HEAD_ID)
				skiplist[prev].next = pos;
			else
				SKIP_HEAD.next = pos;

			// next generate sketches
			int x = pos;
			ArrayList<Integer> L = new ArrayList<Integer>();
			while (x!=MY_NAN && x != TAIL_ID && L.size() < K)//FIX x!=MY_NAN 
			{
				x = skiplist[x].next;
				if (x!=MY_NAN)
					if(x != TAIL_ID || occurrences[x].prev < pos)//FIX x!=MY_NAN
						L.add(x);
			}

			// for (auto rit = L.rbegin(); rit != L.rend(); ++rit)
			for (int i = L.size() - 1; i >= 0; i--) {
				int cx = L.get(i);
				int cy;
				if (cx != TAIL_ID)
					cy = skiplist[cx].prev;
				else
					cy = SKIP_TAIL.prev;

				int D_size = 0;
				int HEAD_REF = MY_NAN;

				while (cy != HEAD_ID && D_size <= K && occurrences[cy].next != pos) {
					if (cy == pos || occurrences[cy].next > cx) {
						D[cy].next = HEAD_REF;
						D[cy].prev = MY_NAN;
						if (HEAD_REF != MY_NAN)
							D[HEAD_REF].prev = cy;
						HEAD_REF = cy;
						D_size += 1;
					} else if (occurrences[cy].next == cx)
						break;
					else if (occurrences[cy].next < cx) {
						int del = occurrences[cy].next;
						if (HEAD_REF == del)
							HEAD_REF = D[del].next;

						if (D[del].next != MY_NAN) {
							int del_next = D[del].next;
							D[del_next].prev = D[del].prev;
						}

						if (D[del].prev != MY_NAN) {
							int del_prev = D[del].prev;
							D[del_prev].next = D[del].next;
						}

						D[cy].next = HEAD_REF;
						D[cy].prev = MY_NAN;
						if (HEAD_REF != MY_NAN)
							D[HEAD_REF].prev = cy;
						HEAD_REF = cy;
					}

					if (D_size == K) {
						int ll = skiplist[cy].prev + 1;
						int lr = cy;
						int rr = cx - 1;
						int index = 0;
						int left_l = 0;
						int right_r = 0;
						while (index < ppos.size()) {
							if (ppos.get(index) <= lr && ll <= ppos.get(index)) {
								left_l = ppos.get(index);
								break;
							}
							index++;
						}
						int idx = HEAD_REF;
						ArrayList<Integer> positions = new ArrayList<Integer>();
						ArrayList<Integer> sketch = new ArrayList<Integer>();
						int rl = 0;
						while (idx != MY_NAN) {
							positions.add(idx);
							sketch.add(hval(hf, doc.get(idx)));
							rl = idx;
							idx = D[idx].next;
						}
						while (index < ppos.size()) {
							if (ppos.get(index) <= rr && rl <= ppos.get(index)) {
								right_r = ppos.get(index);
								break;
							}
							index++;
						}
						// index = 0;
						if (index != ppos.size()) {
							if ((rl - cy) >= tau) {
								res_cws.add(new CompositeWindow(skiplist[cy].prev + 1, cy, rl, cx - 1));
								for (int j = 0; j < K; j++) {
									res_cws.get(res_cws.size() - 1).positions.add(positions.get(j));
									res_cws.get(res_cws.size() - 1).sketch.add(sketch.get(j));
								}
							}
						}
					}
					cy = skiplist[cy].prev;
				}
			}
		}
	}

	private static void generateHashFunc(long seed, ArrayList<Pair> hf) {
		Random rand = new Random(seed);
		int a = 0;
		while (a == 0) {
			a = rand.nextInt(Integer.MAX_VALUE);
		}
		int b = rand.nextInt(Integer.MAX_VALUE);
		hf.add(new Pair(a, b));
	}
}

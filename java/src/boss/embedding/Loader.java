package boss.embedding;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;


public class Loader {
	String trick = "1 1 Καὶ �?γένετο";
	
	static void show_all() {
		// Creating a File object for directory
		File directoryPath = new File(".\\data");
		show_all(directoryPath);
	}

	static void show_all(File directoryPath) {
		System.out.println("List of files and directories in the specified directory:");
		File filesList[] = directoryPath.listFiles();
		for (File file : filesList) {
			System.out.println("File name: " + file.getName());
			if(file.isDirectory()) {
				System.out.println("Is directory, decending ...");
				show_all(file);
			}else if(file.isFile()) {
				//show_one(file);
				clean(file);
			}
		}
	}

	private static void show_one(File file) {
		BufferedReader br;
		int lines_to_display = 100 ;
		
		try {
			br = new BufferedReader(new FileReader(file));
			String line;
			int i = 0;
		
	        while ((line = br.readLine()) != null && i++ < lines_to_display) {
	        	System.out.println(line);
	        	i++;
	        }
	        br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
	
	static ArrayList<String> clean(File file) {
		BufferedReader br;
		
		try {
			br = new BufferedReader(new FileReader(file));
			String line;
			int i = 0;
			String meta_data = br.readLine();
			String[] meta_data_tokens = meta_data.split(" ");
		
			
			ArrayList<String> raw_entries = new ArrayList<String>(10000);//size guessed
			int counter = 0;
	        while ((line = br.readLine()) != null) {
	        	i++;
	        	String[] tokens = line.split(" ");
	        	
	        	if(tokens[0].matches("[a-zA-Z0-9äöüÄÖÜß]*")){
	        		//System.out.println(tokens[0]);
	        		counter++;
	        		raw_entries.add(tokens[0]);
	        	}else{
	        		//System.err.println(tokens[0]);
	        	}
	        	
	        	if(i%100000==0) {
	        		System.out.println(i+" of "+meta_data_tokens[0] +" "+ tokens[0]);	
	        	}
	        }
	        System.out.println(counter+" of "+meta_data_tokens[0] +" valid entries");
	        br.close();
	        return raw_entries;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	static String regex_alphabetic_characters(){
		String alphabetic_characters = "[^0-9a-zA-ZΑ-Ωα-ωäöüÄÖÜßίϊ�?όάέ�?ϋΰήώ ]";
		return alphabetic_characters;
	}

	public static void main(String[] args) {
		//show_all();
		/*String file = ".\\data\\en\\cc.en.300.vec";
		//String file = ".\\data\\de\\cc.de.300.vec";
		Embedding e = new Embedding(file);
		Match[] matches = e.match_all_words(Words.words_en);
		for(Match m : matches) {
			System.out.println(m);
		}*/
		ArrayList<MatchesWithEmbeddings> all_mews = Embedding.get_minimal_embedding_en();
		for(MatchesWithEmbeddings mew : all_mews){
			System.out.println(mew.toTSV());
		}
	}

	public static String get_file(int language) {
		System.err.println("todo");
		return null;
	}
}

package pan;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MatrixLoader {
	static final String path_to_matrices = "./results/pan_results/";
	
	public static List<String> listFilesUsingFilesList(String dir) {
		File directory = new File(dir);
		if(!directory.exists()) {
			System.err.println(dir+" does not exist");
		}
		
	    try (Stream<Path> stream = Files.list(Paths.get(dir))) {
	        return stream
	          .filter(file -> Files.isDirectory(file))
	          .map(Path::getFileName)
	          .map(Path::toString)
	          .collect(Collectors.toList());
	    }catch (Exception e) {
			System.err.println(e);
		}
	    return null;
	}
	
	public static void main(String[] args) {
		/*List<String> directories = listFilesUsingFilesList(path_to_matrices);
		for(String dir : directories) {
			System.out.println(dir);
		}
		System.out.println("get_all_susp_src_directories()");
		for(String dir : get_all_susp_src_directories(directories)) {
			System.out.println(dir);
		}
		System.out.println("get_all_excerpt_directories()");
		for(String dir : get_all_excerpt_directories(directories)) {
			System.out.println(dir);
		}*/
		//load_all_matrices();
		//load_all__excerpt_matrices();
		//get_org_docs_and_excerpts(0);
		PotthastMetrics.run();
		//PotthastMetrics.run_full_documents();
	}
	
	static List<String> get_all_susp_src_directories(){
		return get_all_susp_src_directories(listFilesUsingFilesList(path_to_matrices)); 
	}
	
	static List<String> get_all_susp_src_directories(List<String> all_directories){
		ArrayList<String> all_excerpt_directories = new ArrayList<String>();
		for(String s : all_directories) {
			if(s.startsWith("susp")) {
				all_excerpt_directories.add(s);
			}
		}
		return all_excerpt_directories;
	}
	
	static List<String> get_all_excerpt_directories(){
		return get_all_excerpt_directories(listFilesUsingFilesList(path_to_matrices));
	}
	
	static List<String> get_all_excerpt_directories(List<String> all_directories){
		ArrayList<String> all_excerpt_directories = new ArrayList<String>();
		for(String s : all_directories) {
			if(!s.startsWith("susp")) {
				all_excerpt_directories.add(s);
			}
		}
		return all_excerpt_directories;
	}
	
	/**
	 * return all excerpt matrices and their full text counterparts
	 */
	public static ArrayList<double[][]>[] get_org_docs_and_excerpts(int file_num) {
		List<String> s = get_all_excerpt_directories(listFilesUsingFilesList(path_to_matrices));
		
		String dir = s.get(file_num);
		ArrayList<double[][]> excerpts = load_all_matrices_of_pair(dir);
		String org_doc_dir = get_org_document_dir(dir);
		ArrayList<double[][]> org_documents = load_all_matrices_of_pair(org_doc_dir);
		
		ArrayList<double[][]>[] ret = new ArrayList[2];
		ret[0] = excerpts;
		ret[1] = org_documents;
		return ret;
	}
		
	/**
	 * 
	 * @param dir input like "00228 excerpt [1925,2445]_05889 excerpt [581,1070]"
	 * @return
	 */
	public static String get_org_document_dir(String dir) {
		String[] ids = dir.split("_");
		String first_id_str = ids[0].substring(0, 5);
		String second_id_str = ids[1].substring(0, 5);
		String org_document_dir = "susp_"+first_id_str+"_src_"+second_id_str;
		return org_document_dir;
	}
	/**
	 * 
	 * @param dir input like "00228 excerpt [1925,2445]_05889 excerpt [581,1070]"
	 * @return
	 */
	static String[] get_document_ids(String dir) {
		String[] ids = dir.split("_");
		String first_id_str = ids[0].substring(0, 5);
		String second_id_str = ids[1].substring(0, 5);
		String[] ret = {first_id_str, second_id_str};
		return ret;
	}

	static double[][] load(File f) {
		double start = System.currentTimeMillis();
		ArrayList<double[]> temp = new ArrayList<double[]>();
		try {
			BufferedReader reader = new BufferedReader(new FileReader(f));
			String line;
			
			while((line = reader.readLine()) != null) {
				String[] line_values = line.split("\t");
				double[] line_as_double = new double[line_values.length];
				for(int i=0;i<line_values.length;i++) {
					String s = line_values[i];
					double d = Double.parseDouble(s);
					line_as_double[i] = d;
				}
				temp.add(line_as_double);
			}
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		double[][] ret = new double[temp.size()][];
		for(int i=0;i<temp.size();i++) {
			ret[i] = temp.get(i);
		}
		System.out.println("Loaded matrix of size "+ret.length+" * "+ret[0].length+" in "+(System.currentTimeMillis()-start)+" ms from "+f);
		return ret;
	}
	public static ArrayList<double[][]> load_all_matrices_of_pair(String f) {
		int[] k_s = {3,4,5,6,7,8};
		return load_all_matrices_of_pair(new File(f), k_s) ;
	}
	static ArrayList<double[][]> load_all_matrices_of_pair(File f) {
		int[] k_s = {3,4,5,6,7,8};
		return load_all_matrices_of_pair(f, k_s) ;
	}
	static ArrayList<double[][]> load_all_matrices_of_pair(File dir, int[] k_s) {
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		for(int k : k_s) {
			File f = new File(path_to_matrices+"/"+dir+"/"+k+".tsv");
			if(!f.exists()) {
				System.err.println(f+" does not exist");
			}else {
				double[][] matrix = load(f);
				ret.add(matrix);
			}
		}
		return ret;
	}
	static void load_all_matrices() {
		for(String dir : listFilesUsingFilesList(path_to_matrices)) {
			load_all_matrices_of_pair(dir);
		}
	}
	public static ArrayList<ArrayList<double[][]>> load_all_excerpt_matrices() {
		ArrayList<ArrayList<double[][]>> all_matrices = new ArrayList<ArrayList<double[][]>>();
		for(String dir : get_all_excerpt_directories(listFilesUsingFilesList(path_to_matrices))) {
			ArrayList<double[][]> pair_matrices = load_all_matrices_of_pair(dir);
			all_matrices.add(pair_matrices);
		}
		return all_matrices;
	}
	public static ArrayList<ArrayList<double[][]>> load_all_full_document_matrices() {
		ArrayList<ArrayList<double[][]>> all_matrices = new ArrayList<ArrayList<double[][]>>();
		for(String dir : get_all_susp_src_directories(listFilesUsingFilesList(path_to_matrices))) {
			ArrayList<double[][]> pair_matrices = load_all_matrices_of_pair(dir);
			all_matrices.add(pair_matrices);
		}
		return all_matrices;
	}
}

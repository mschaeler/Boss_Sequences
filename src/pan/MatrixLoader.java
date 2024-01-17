package pan;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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
		List<String> directories = listFilesUsingFilesList(path_to_matrices);
		for(String dir : directories) {
			System.out.println(dir);
		}
	}
}

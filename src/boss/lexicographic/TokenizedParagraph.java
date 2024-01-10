package boss.lexicographic;

import java.util.ArrayList;

import plus.data.Chapter;
import plus.data.Paragraph;

/**
 * Container that gives access to all processing steps when turning a raw text paragraph into lexicographic Tokens. 
 * Is supposed to be Input for SemanticTokenized paragraphs using last_intermediate_result()
 */
public class TokenizedParagraph extends Paragraph{

	String[] org_paragraph;
	ArrayList<String[]> inter_mediate_results=new ArrayList<String[]>();
	ArrayList<String> step_name = new ArrayList<String>();
	
	public TokenizedParagraph(Chapter _c, String _paragraph_name, String _paragraph, Tokenizer t) {
		super(_c, _paragraph_name, _paragraph);
		this.org_paragraph = t.basic_tokenization(_paragraph);
	}
	
	/**
	 * This is the interface to subsequent parts of the infrastructure
	 * @return
	 */
	public ArrayList<StringToken> get_tokens(){
		sanity_check();
		String final_result[] = inter_mediate_results.get(inter_mediate_results.size()-1);//The last one is supposed to be the final result
		ArrayList<StringToken> ret = new ArrayList<StringToken>(final_result.length);
		for(String s : final_result) {
			if(s!=null) {
				ret.add(new StringToken(s));
			}
		}
		
		return ret;
	}
	/**
	 * We assume that the length of all intermediate result is the same, i.e., it may contain null values. Only calling get_tokens() creates a dense List of Tokens.
	 * @return
	 */
	boolean sanity_check() {
		final int length = org_paragraph.length;
		for(int i=0;i<inter_mediate_results.size();i++) {
			String[] temp = inter_mediate_results.get(i);
			if(temp.length!=length){
				System.err.println("TokenizedParagraph.sanity_check() length != length@"+i);
				return false;
			}
		}
		return true;
	}
	
	public String toString() {
		String ret = "Paragraph "+paragraph_name;
		ret += "\norg\t"+my_paragraph_as_text;
		for(int i=0;i<inter_mediate_results.size();i++) {//TODO step names
			String[] array = inter_mediate_results.get(i);
			//ret+="\n"+this.step_name.get(i)+"\t"+outTSV(array);
			ret+="\n"+step_name.get(i).substring(0, 7)+"\t"+outTSV(array);
		}
		return ret;
	}
	String outTSV(String[] array){
		String ret = array[0];
		for(int i=1;i<array.length;i++) {
			ret+="\t"+array[i];
		}
		return ret;
	}

	public String[] last_intermediate_result() {
		if(this.inter_mediate_results.isEmpty()) {
			return this.org_paragraph;
		}else{
			return inter_mediate_results.get(inter_mediate_results.size() - 1);
		}
	}
	
	public int size() {
		return get_tokens().size();
	}
}

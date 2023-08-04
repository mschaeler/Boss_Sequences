package boss.token_mapping;

import java.util.ArrayList;
import java.util.HashMap;

import boss.lexicographic.StringToken;
import boss.semantic.SemanticToken;
import boss.semantic.SetSemanticToken;

public class SetSemantic extends MyDictionary{

	private HashMap<String,ArrayList<String>> dict_entries = null;
	
	public void set_emebdding(HashMap<String,ArrayList<String>> dict_entries) {
		this.dict_entries = dict_entries;
	}
	
	@Override
	public SemanticToken map_to_semantic(StringToken t) {
		SetSemanticToken sst = new SetSemanticToken(t);//only fullform
		//determine baseform
		String baseform = map_to_word(t);
		sst.baseform = baseform;
		//determine semantic
		if(dict_entries != null) {
			ArrayList<String> dict_entries = this.dict_entries.get(baseform);
			if(dict_entries==null) {
				//should never happen
				System.err.println("SetSemantic.map_to_semantic() dict_entries==null");
			}
			sst.semantics = dict_entries;
		}else {
			System.out.println("Warning: SetSemantic.map_to_semantic() embedding = null");
		}
		return sst;
	}

}

//
// Created by Martin on 17.07.2023.
//

#ifndef PRANAY_TEST_ENVIRONMENT_H
#define PRANAY_TEST_ENVIRONMENT_H

#include <string>
#include <utility>
#include <vector>
#include <fstream>      // std::ifstream
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <regex>

using namespace std;

vector<string> DONG_DENG_STOPWORDS = {
        "i"
        ,"me"
        ,"1"
        ,"2"
        ,"3"
        ,"4"
        ,"5"
        ,"6"
        ,"7"
        ,"8"
        ,"9"
        ,"0"
        ,"my"
        ,"myself"
        ,"we"
        ,"our"
        ,"ours"
        ,"ourselves"
        ,"you"
        ,"your"
        ,"yours"
        ,"yourself"
        ,"yourselves"
        ,"he"
        ,"him"
        ,"his"
        ,"himself"
        ,"she"
        ,"her"
        ,"hers"
        ,"herself"
        ,"it"
        ,"its"
        ,"itself"
        ,"they"
        ,"them"
        ,"their"
        ,"theirs"
        ,"themselves"
        ,"what"
        ,"which"
        ,"who"
        ,"whom"
        ,"this"
        ,"that"
        ,"these"
        ,"those"
        ,"am"
        ,"is"
        ,"are"
        ,"was"
        ,"were"
        ,"be"
        ,"been"
        ,"being"
        ,"have"
        ,"has"
        ,"had"
        ,"having"
        ,"do"
        ,"does"
        ,"did"
        ,"doing"
        ,"a"
        ,"an"
        ,"the"
        ,"and"
        ,"but"
        ,"if"
        ,"or"
        ,"because"
        ,"as"
        ,"until"
        ,"while"
        ,"of"
        ,"at"
        ,"by"
        ,"for"
        ,"with"
        ,"about"
        ,"against"
        ,"between"
        ,"into"
        ,"through"
        ,"during"
        ,"before"
        ,"after"
        ,"above"
        ,"below"
        ,"to"
        ,"from"
        ,"up"
        ,"down"
        ,"in"
        ,"out"
        ,"on"
        ,"off"
        ,"over"
        ,"under"
        ,"again"
        ,"further"
        ,"then"
        ,"once"
        ,"here"
        ,"there"
        ,"when"
        ,"where"
        ,"why"
        ,"how"
        ,"all"
        ,"any"
        ,"both"
        ,"each"
        ,"few"
        ,"more"
        ,"most"
        ,"other"
        ,"some"
        ,"such"
        ,"no"
        ,"nor"
        ,"not"
        ,"only"
        ,"own"
        ,"same"
        ,"so"
        ,"than"
        ,"too"
        ,"very"
        ,"s"
        ,"t"
        ,"can"
        ,"will"
        ,"just"
        ,"don"
        ,"should"
        ,"now"
};

vector<string> GERMAN_STOP_WORDS = {
        "aber"
        , "als"
        , "am"
        , "an"
        , "auch"
        , "auf"
        , "aus"
        , "bei"
        , "bin"
        , "bis"
        , "bist"
        , "da"
        , "dadurch"
        , "daher"
        , "darum"
        , "das"
        , "daß"
        , "dass"
        , "dein"
        , "deine"
        , "dem"
        , "den"
        , "der"
        , "des"
        , "dessen"
        , "deshalb"
        , "die"
        , "dies"
        , "dieser"
        , "dieses"
        , "doch"
        , "dort"
        , "du"
        , "durch"
        , "ein"
        , "eine"
        , "einem"
        , "einen"
        , "einer"
        , "eines"
        , "er"
        , "es"
        , "euer"
        , "eure"
        , "für"
        , "hatte"
        , "hatten"
        , "hattest"
        , "hattet"
        , "hier"
        , "hinter"
        , "ich"
        , "ihr"
        , "ihre"
        , "im"
        , "in"
        , "ist"
        , "ja"
        , "jede"
        , "jedem"
        , "jeden"
        , "jeder"
        , "jedes"
        , "jener"
        , "jenes"
        , "jetzt"
        , "kann"
        , "kannst"
        , "können"
        , "könnt"
        , "machen"
        , "mein"
        , "meine"
        , "mit"
        , "muß"
        , "mußt"
        , "musst"
        , "müssen"
        , "müßt"
        , "nach"
        , "nachdem"
        , "nein"
        , "nicht"
        , "nun"
        , "oder"
        , "seid"
        , "sein"
        , "seine"
        , "sich"
        , "sie"
        , "sind"
        , "soll"
        , "sollen"
        , "sollst"
        , "sollt"
        , "sonst"
        , "soweit"
        , "sowie"
        , "und"
        , "unser"
        , "unsere"
        , "unter"
        , "vom"
        , "von"
        , "vor"
        , "wann"
        , "warum"
        , "was"
        , "weiter"
        , "weitere"
        , "wenn"
        , "wer"
        , "werde"
        , "werden"
        , "werdet"
        , "weshalb"
        , "wie"
        , "wieder"
        , "wieso"
        , "wir"
        , "wird"
        , "wirst"
        , "wo"
        , "woher"
        , "wohin"
        , "zu"
        , "zum"
        , "zur"
        , "über"
};

class Environment{
private:
    std::unordered_map<int, vector<int>> sets;
    set<int> text1sets;
    set<int> text2sets;
    std::unordered_set<int> wordSet;
    vector<set<int>> invertedIndex;
    std::unordered_map<int, string> int2word;
    std::unordered_map<string, int> word2int;

    int language = EN;

    static bool is_alnum_or_space_de(const char c){
        string special_chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß";
        return (special_chars.find(c) == string::npos);
    }

    static bool is_alnum_or_space(const char c){
        string special_chars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";//äöüÄÖÜß
        return (special_chars.find(c) == string::npos);
    }

    template <typename Out>
    static void split(const std::string &s, char delim, Out result) {
        std::istringstream iss(s);
        std::string item;
        while (std::getline(iss, item, delim)) {
            *result++ = item;
        }
    }

    static std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> elems;
        split(s, delim, std::back_inserter(elems));
        return elems;
    }

    static vector<string> remove_stopwords(const vector<string>& tokens, const vector<string>& stop_words) {
        vector<string> result;
        for(const auto& t : tokens){
            if(find(stop_words.begin(), stop_words.end(), t) == stop_words.end()){//is not a stop word
                if(!t.empty()) {
                    result.push_back(t);
                }
            }
        }
        return result;
    }


    void do_crazy_things_2(vector<string>& text_files, set<int>& text_sets){
        for (const auto& f : text_files) {//For each text file containing a single word
            cout << "Loading File " << f << endl;
            //set2int[f] = text_id;
            //int2set[text_id] = f;
            text_sets.insert(text_id);
            string line;
            ifstream infile(f);
            if (infile.is_open()) {
                cout << "Skipping line: ";
                getline(infile, line);//Name of the Biblical Book
                cout << line << endl;

                while (getline(infile, line)) {
                    if(line.rfind("--",0)!=0){//Begin of chapter marker
                        vector<string> tokens = tokenize(line, language);
                        for(const string& token : tokens){
                            if (!token.empty()) {
                                if (word2int.find(token) == word2int.end()) {
                                    word2int[token] = id;
                                    wordSet.insert(id);
                                    int2word[id] = token;
                                    sets[text_id].push_back(id);
                                    std::set<int> nset = {text_id};
                                    invertedIndex.push_back(nset);
                                    id += 1;
                                } else {
                                    sets[text_id].push_back(word2int[token]);
                                    invertedIndex[word2int[token]].insert(text_id);
                                }
                            }
                        }
                    }else{
                        cout << "Skipping line: " << line << endl;
                    }
                }
            }else{
                cout << "Could not open " << f << endl;
            }
            text_id += 1;
        }
    }

    void register_token(const string& token){
        if (!token.empty()) {
            if (word2int.find(token) == word2int.end()) {
                word2int[token] = id;
                wordSet.insert(id);
                int2word[id] = token;
                sets[text_id].push_back(id);
                std::set<int> nset = {text_id};
                invertedIndex.push_back(nset);
                id += 1;
            } else {
                sets[text_id].push_back(word2int[token]);
                invertedIndex[word2int[token]].insert(text_id);
            }
        }
    }

public:
    int id = 0;
    //int tokens = 0;
    int text_id = 0;
    const static int EN = 0;
    const static int DE = 1;

    Environment(const vector<string>& all_tokens, int length){//copy the token vector
        //reduce to length than split in half
        text_id = 0;
        for(int i=0;i<length/2;i++){
            register_token(all_tokens.at(i));
        }
        text_id = 1;
        for(int i=length/2;i<length;i++){
            register_token(all_tokens.at(i));
        }
    }

    Environment(string text1location, string text2location) {
        vector<string> text1_files = {std::move(text1location)};
        vector<string> text2_files = {std::move(text2location)};

        cout << text1_files.size() << " text1_files listed" << endl;
        cout << text2_files.size() << " text2_files listed" << endl;

        do_crazy_things_2(text1_files, text1sets);
        do_crazy_things_2(text2_files, text2sets);
    }

    Environment(string text1location, string text2location, int _language) : language(_language) {
        vector<string> text1_files = {std::move(text1location)};
        vector<string> text2_files = {std::move(text2location)};

        cout << text1_files.size() << " text1_files listed" << endl;
        cout << text2_files.size() << " text2_files listed" << endl;

        do_crazy_things_2(text1_files, text1sets);
        do_crazy_things_2(text2_files, text2sets);
    }

    void out(){
        cout << "<int, vector<int>> sets" << endl;
        for(const auto& s : sets){
            cout << s.first << "={";
            for(auto v : s.second){
                cout << v << " ";
            }
            cout << "}" << endl;
        }
        for(const auto& s : sets){
            cout << s.first << "={";
            for(auto v : s.second){
                cout << toWord(v) << " ";
            }
            cout << "}" << endl;
        }

        cout << "vector<int> text1sets" << endl;
        for(auto i : text1sets){
            cout << i << " ";
        }
        cout << endl;

        cout << "vector<int> text2sets" << endl;
        for(auto i : text2sets){
            cout << i << " ";
        }
        cout << endl;
    }

    std::unordered_map<int, vector<int>> getSets() {
        return sets;
    }

    int toInt(const string& t) {
        if (word2int.find(t) == word2int.end()) {
            return -1;
        } else {
            return word2int[t];
        }
    }

    string toWord(int i) {
        return int2word[i];
    }

    set<int>& getText1SetIds() {
        return text1sets;
    }

    set<int>& getText2SetIds() {
        return text2sets;
    }

    static vector<string> tokenize(string line, int language){
        //cout << "org= " << line << endl;
        // remove non-alphabetic chars
        if(language==DE) {
            line.erase(std::remove_if(line.begin(), line.end(), is_alnum_or_space_de), line.end());
        }else{
            line.erase(std::remove_if(line.begin(), line.end(), is_alnum_or_space), line.end());
        }
        //cout << "alphanumeric= " << line << endl;
        // replace duplicate white spaces
        regex reg(" +");
        line = std::regex_replace(line,  reg, " ");
        //cout << "no dupl white spaces= " << line << endl;
        // to lower case
        std::transform(line.begin(), line.end(), line.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        //Split into string tokens
        //cout << "lower case= " << line << endl;
        vector<string> tokens = split(line, ' ');
        /*for(string s : tokens){
            cout << s << " ";
        }
        cout << endl;*/
        //Remove stop words
        if(language==DE) {
            tokens = remove_stopwords(tokens, GERMAN_STOP_WORDS);
        }else{
            tokens = remove_stopwords(tokens, DONG_DENG_STOPWORDS);
        }

        /*for(const auto& s : tokens){
            cout << s << "\t";
        }
        cout << endl;*/
        for(const auto& t : tokens){
            if(t.empty()){
                cout << "Empty string found" << endl;
            }
        }
        return tokens;
    }
};
#endif //PRANAY_TEST_ENVIRONMENT_H

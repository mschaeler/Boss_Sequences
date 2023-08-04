//
// Created by Martin on 17.07.2023.
// Edited by Pranay on 04.08.2023.
//

#ifndef PRANAY_TEST_ENVIRONMENT_H
#define PRANAY_TEST_ENVIRONMENT_H

#include <string>
#include <vector>
#include <fstream>      // std::ifstream
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

class Environment{
private:
    std::unordered_map<int, vector<int>> sets;
    set<int> text1sets;
    set<int> text2sets;
    std::unordered_set<int> wordSet;
    vector<set<int>> invertedIndex;
    std::unordered_map<int, string> int2word;
    std::unordered_map<string, int> word2int;
    std::unordered_map<int, string> int2set;
    std::unordered_map<string, int> set2int;
    int text1_average_cardinality = 0;
    int text2_average_cardinality = 0;

    vector<std::vector<int>> slidingWindows(vector<int>& nums, int k) {
        std::vector<std::vector<int>> result;
        if (nums.empty() || k <= 0 || k > nums.size()) {
            return result;
        }

        auto it = nums.begin();
        std::vector<int> window(it, std::next(it, k));
        result.push_back(window);

        while (std::next(it, k) != nums.end()) {
            window.erase(window.begin());
            window.push_back(*std::next(it, k));
            result.push_back(window);
            ++it;
        }

        return result;
    }

public:
    Environment(string text1location, string text2location){

        //get_files_from_directory(text1location);
        cout << "creating environment with lakes: " << text1location << ", " << text2location << endl;
        vector<string> text1_files;
        vector<string> text2_files;

        for (const auto &entry: fs::directory_iterator(text1location)) {
            text1_files.push_back(entry.path());
        }

        for (const auto &entry: fs::directory_iterator(text2location)) {
            text2_files.push_back(entry.path());
        }

        cout << text1_files.size() << " text1_files listed" << endl;
        cout << text2_files.size() << " text2_files listed" << endl;

        int id = 0;
        int tokens = 0;
        int sid = 0;

        for (size_t i = 0; i < text1_files.size(); ++i) {
            string f = text1_files[i];
            //cout << "File " << f << endl;
            set2int[f] = sid;
            int2set[sid] = f;
            text1sets.insert(sid);
            string line;
            ifstream infile(f);
            if (infile.is_open()) {
                string line;
                while (getline(infile, line)) {
                    //cout << line << endl;
                    tokens += 1;
                    line.erase(line.find_last_not_of(" \n\r\t")+1);
                    if (line.size() > 0) {
                        if (word2int.find(line) == word2int.end()) {
                            word2int[line] = id;
                            wordSet.insert(id);
                            int2word[id] = line;
                            sets[sid].push_back(id);
                            std::set<int> nset = {sid};
                            invertedIndex.push_back(nset);
                            id += 1;
                        } else {
                            sets[sid].push_back(word2int[line]);
                            invertedIndex[word2int[line]].insert(sid);
                        }
                    }
                }
            }else{
                cout << "Could not open " << f << endl;
            }
            sid += 1;
        }

        int text1_tokens = tokens;
        int text1_sets = sets.size();
        text1_average_cardinality = text1_tokens / text1_sets;

        for (size_t i = 0; i < text2_files.size(); ++i) {
            string f = text2_files[i];
            //cout << "File " << f << endl;
            set2int[f] = sid;
            int2set[sid] = f;
            text2sets.insert(sid);
            ifstream infile(f);
            if (infile.is_open()) {
                string line;
                while (getline(infile, line)) {
                    tokens += 1;
                    line.erase(line.find_last_not_of(" \n\r\t")+1);
                    // cout << "Line for File: " << f << " " << line << endl;
                    if (line.size() > 0) {
                        if (word2int.find(line) == word2int.end()) {
                            word2int[line] = id;
                            wordSet.insert(id);
                            int2word[id] = line;
                            sets[sid].push_back(id);
                            std::set<int> nset = {sid};
                            invertedIndex.push_back(nset);
                            id += 1;
                        } else {
                            sets[sid].push_back(word2int[line]);
                            invertedIndex[word2int[line]].insert(sid);
                        }
                    }
                }
            }else{
                cout << "Could not open " << f << endl;
            }
            sid += 1;

        }

        text2_average_cardinality = (tokens - text1_tokens) / (sets.size() - text1_sets);
    }

    void out(){
        cout << "<int, vector<int>> sets" << endl;
        for(auto s : sets){
            cout << s.first << "={";
            for(auto v : s.second){
                cout << v << " ";
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

    std::unordered_map<int, vector<vector<int>>> computeSlidingWindows(int windowWidth) {
        std::unordered_map<int, vector<vector<int>>> windows;
        for (auto i = sets.begin(); i != sets.end(); i++) {
            windows[i->first] = slidingWindows(i->second, windowWidth);
        }
        return windows;
    }

    std::unordered_set<int>& getWordSet() {
        return wordSet;
    }

    vector<set<int>>& getInvertedIndex() {
        return invertedIndex;
    }

    int getText1Avg() {
        return text1_average_cardinality;
    }

    int getText2Avg() {
        return text2_average_cardinality;
    }

    int toInt(string t) {
        if (word2int.find(t) == word2int.end()) {
            return -1;
        } else {
            return word2int[t];
        }
    }

    string toWord(int i) {
        return int2word[i];
    }

    int getSetId(string s) {
        if (set2int.find(s) == set2int.end()) {
            return -1;
        } else {
            return set2int[s];
        }
    }

    string getSetName(int i) {
        return int2set[i];
    }

    set<int>& getText1SetIds() {
        return text1sets;
    }

    set<int>& getText2SetIds() {
        return text2sets;
    }
};

#endif //PRANAY_TEST_ENVIRONMENT_H

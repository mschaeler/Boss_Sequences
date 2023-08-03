//
// Created by Martin on 17.07.2023.
//

#ifndef PRANAY_TEST_ENVIRONMENT_H
#define PRANAY_TEST_ENVIRONMENT_H

#include <string>
#include <vector>
#include <fstream>      // std::ifstream
#include <set>
#include <unordered_map>
#include <unordered_set>

using namespace std;

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
    Environment(){
        // string text1location = "../data/en/esv/";
        // string text2location = "../data/en/king_james_bible/";
        string text1location = "/root/data/en/esv_book_martin/";
        string text2location = "/root/data/en/king_james_bible_book_martin/";

        //get_files_from_directory(text1location);

        vector<string> text1_files;
        vector<string> text2_files;

        //XXX - MINGW has issues with filesystem include.... Do it by hand
        {
            text1_files.push_back(text1location+"1_1.txt");
            text1_files.push_back(text1location+"1_2.txt");
            text1_files.push_back(text1location+"1_3.txt");
            text1_files.push_back(text1location+"1_4.txt");
            text1_files.push_back(text1location+"1_5.txt");
            text1_files.push_back(text1location+"1_6.txt");
            text1_files.push_back(text1location+"1_7.txt");
            text1_files.push_back(text1location+"1_8.txt");
            text1_files.push_back(text1location+"1_9.txt");
            text1_files.push_back(text1location+"1_10.txt");
            text1_files.push_back(text1location+"1_11.txt");
            text1_files.push_back(text1location+"1_12.txt");
            text1_files.push_back(text1location+"1_13.txt");
            text1_files.push_back(text1location+"1_14.txt");
            text1_files.push_back(text1location+"1_15.txt");
            text1_files.push_back(text1location+"1_16.txt");
            text1_files.push_back(text1location+"1_17.txt");
            text1_files.push_back(text1location+"1_18.txt");
            text1_files.push_back(text1location+"1_19.txt");
            text1_files.push_back(text1location+"1_20.txt");
            text1_files.push_back(text1location+"1_21.txt");
            text1_files.push_back(text1location+"1_22.txt");
            text1_files.push_back(text1location+"2_1.txt");
            text1_files.push_back(text1location+"2_2.txt");
            text1_files.push_back(text1location+"2_3.txt");
            text1_files.push_back(text1location+"2_4.txt");
            text1_files.push_back(text1location+"2_5.txt");
            text1_files.push_back(text1location+"2_6.txt");
            text1_files.push_back(text1location+"2_7.txt");
            text1_files.push_back(text1location+"2_8.txt");
            text1_files.push_back(text1location+"2_9.txt");
            text1_files.push_back(text1location+"2_10.txt");
            text1_files.push_back(text1location+"2_11.txt");
            text1_files.push_back(text1location+"2_12.txt");
            text1_files.push_back(text1location+"2_13.txt");
            text1_files.push_back(text1location+"2_14.txt");
            text1_files.push_back(text1location+"2_15.txt");
            text1_files.push_back(text1location+"2_16.txt");
            text1_files.push_back(text1location+"2_17.txt");
            text1_files.push_back(text1location+"2_18.txt");
            text1_files.push_back(text1location+"2_19.txt");
            text1_files.push_back(text1location+"2_20.txt");
            text1_files.push_back(text1location+"2_21.txt");
            text1_files.push_back(text1location+"2_22.txt");
            text1_files.push_back(text1location+"2_23.txt");
            text1_files.push_back(text1location+"3_1.txt");
            text1_files.push_back(text1location+"3_2.txt");
            text1_files.push_back(text1location+"3_3.txt");
            text1_files.push_back(text1location+"3_4.txt");
            text1_files.push_back(text1location+"3_5.txt");
            text1_files.push_back(text1location+"3_6.txt");
            text1_files.push_back(text1location+"3_7.txt");
            text1_files.push_back(text1location+"3_8.txt");
            text1_files.push_back(text1location+"3_9.txt");
            text1_files.push_back(text1location+"3_10.txt");
            text1_files.push_back(text1location+"3_11.txt");
            text1_files.push_back(text1location+"3_12.txt");
            text1_files.push_back(text1location+"3_13.txt");
            text1_files.push_back(text1location+"3_14.txt");
            text1_files.push_back(text1location+"3_15.txt");
            text1_files.push_back(text1location+"4_1.txt");
            text1_files.push_back(text1location+"4_2.txt");
            text1_files.push_back(text1location+"4_3.txt");
            text1_files.push_back(text1location+"4_4.txt");
            text1_files.push_back(text1location+"4_5.txt");
            text1_files.push_back(text1location+"4_6.txt");
            text1_files.push_back(text1location+"4_7.txt");
            text1_files.push_back(text1location+"4_8.txt");
            text1_files.push_back(text1location+"4_9.txt");
            text1_files.push_back(text1location+"4_10.txt");
            text1_files.push_back(text1location+"4_11.txt");
            text1_files.push_back(text1location+"4_12.txt");
            text1_files.push_back(text1location+"4_13.txt");
            text1_files.push_back(text1location+"4_14.txt");
            text1_files.push_back(text1location+"4_15.txt");
            text1_files.push_back(text1location+"4_16.txt");
            text1_files.push_back(text1location+"4_17.txt");
            text1_files.push_back(text1location+"5_1.txt");
            text1_files.push_back(text1location+"5_2.txt");
            text1_files.push_back(text1location+"5_3.txt");
            text1_files.push_back(text1location+"5_4.txt");
            text1_files.push_back(text1location+"5_5.txt");
            text1_files.push_back(text1location+"5_6.txt");
            text1_files.push_back(text1location+"5_7.txt");
            text1_files.push_back(text1location+"5_8.txt");
            text1_files.push_back(text1location+"5_9.txt");
            text1_files.push_back(text1location+"5_10.txt");
            text1_files.push_back(text1location+"5_11.txt");
            text1_files.push_back(text1location+"5_12.txt");
            text1_files.push_back(text1location+"5_13.txt");
            text1_files.push_back(text1location+"5_14.txt");
            text1_files.push_back(text1location+"6_1.txt");
            text1_files.push_back(text1location+"6_2.txt");
            text1_files.push_back(text1location+"6_3.txt");
            text1_files.push_back(text1location+"6_4.txt");
            text1_files.push_back(text1location+"6_5.txt");
            text1_files.push_back(text1location+"6_6.txt");
            text1_files.push_back(text1location+"6_7.txt");
            text1_files.push_back(text1location+"6_8.txt");
            text1_files.push_back(text1location+"6_9.txt");
            text1_files.push_back(text1location+"6_10.txt");
            text1_files.push_back(text1location+"6_11.txt");
            text1_files.push_back(text1location+"6_12.txt");
            text1_files.push_back(text1location+"6_13.txt");
            text1_files.push_back(text1location+"6_14.txt");
            text1_files.push_back(text1location+"7_1.txt");
            text1_files.push_back(text1location+"7_2.txt");
            text1_files.push_back(text1location+"7_3.txt");
            text1_files.push_back(text1location+"7_4.txt");
            text1_files.push_back(text1location+"7_5.txt");
            text1_files.push_back(text1location+"7_6.txt");
            text1_files.push_back(text1location+"7_7.txt");
            text1_files.push_back(text1location+"7_8.txt");
            text1_files.push_back(text1location+"7_9.txt");
            text1_files.push_back(text1location+"7_10.txt");
            text1_files.push_back(text1location+"8_1.txt");
            text1_files.push_back(text1location+"8_2.txt");
            text1_files.push_back(text1location+"8_3.txt");
            text1_files.push_back(text1location+"8_4.txt");
            text1_files.push_back(text1location+"8_5.txt");
            text1_files.push_back(text1location+"8_6.txt");
            text1_files.push_back(text1location+"8_7.txt");
            text1_files.push_back(text1location+"8_8.txt");
            text1_files.push_back(text1location+"8_9.txt");
            text1_files.push_back(text1location+"8_10.txt");
            text1_files.push_back(text1location+"8_11.txt");
            text1_files.push_back(text1location+"8_12.txt");
            text1_files.push_back(text1location+"8_13.txt");
            text1_files.push_back(text1location+"8_14.txt");
            text1_files.push_back(text1location+"8_15.txt");
            text1_files.push_back(text1location+"8_16.txt");
            text1_files.push_back(text1location+"8_17.txt");
            text1_files.push_back(text1location+"9_1.txt");
            text1_files.push_back(text1location+"9_2.txt");
            text1_files.push_back(text1location+"9_3.txt");
            text1_files.push_back(text1location+"9_4.txt");
            text1_files.push_back(text1location+"9_5.txt");
            text1_files.push_back(text1location+"9_6.txt");
            text1_files.push_back(text1location+"9_7.txt");
            text1_files.push_back(text1location+"9_8.txt");
            text1_files.push_back(text1location+"9_9.txt");
            text1_files.push_back(text1location+"9_10.txt");
            text1_files.push_back(text1location+"9_11.txt");
            text1_files.push_back(text1location+"9_12.txt");
            text1_files.push_back(text1location+"9_13.txt");
            text1_files.push_back(text1location+"9_14.txt");
            text1_files.push_back(text1location+"9_15.txt");
            text1_files.push_back(text1location+"9_16.txt");
            text1_files.push_back(text1location+"9_17.txt");
            text1_files.push_back(text1location+"9_18.txt");
            text1_files.push_back(text1location+"9_19.txt");
            text1_files.push_back(text1location+"9_20.txt");
            text1_files.push_back(text1location+"9_21.txt");
            text1_files.push_back(text1location+"9_22.txt");
            text1_files.push_back(text1location+"9_23.txt");
            text1_files.push_back(text1location+"9_24.txt");
            text1_files.push_back(text1location+"9_25.txt");
            text1_files.push_back(text1location+"9_26.txt");
            text1_files.push_back(text1location+"9_27.txt");
            text1_files.push_back(text1location+"9_28.txt");
            text1_files.push_back(text1location+"9_29.txt");
            text1_files.push_back(text1location+"9_30.txt");
            text1_files.push_back(text1location+"9_31.txt");
            text1_files.push_back(text1location+"9_32.txt");
            text1_files.push_back(text1location+"10_1.txt");
            text1_files.push_back(text1location+"10_2.txt");
            text1_files.push_back(text1location+"10_3.txt");
        }

        {
            text2_files.push_back(text2location+"1_1.txt");
            text2_files.push_back(text2location+"1_2.txt");
            text2_files.push_back(text2location+"1_3.txt");
            text2_files.push_back(text2location+"1_4.txt");
            text2_files.push_back(text2location+"1_5.txt");
            text2_files.push_back(text2location+"1_6.txt");
            text2_files.push_back(text2location+"1_7.txt");
            text2_files.push_back(text2location+"1_8.txt");
            text2_files.push_back(text2location+"1_9.txt");
            text2_files.push_back(text2location+"1_10.txt");
            text2_files.push_back(text2location+"1_11.txt");
            text2_files.push_back(text2location+"1_12.txt");
            text2_files.push_back(text2location+"1_13.txt");
            text2_files.push_back(text2location+"1_14.txt");
            text2_files.push_back(text2location+"1_15.txt");
            text2_files.push_back(text2location+"1_16.txt");
            text2_files.push_back(text2location+"1_17.txt");
            text2_files.push_back(text2location+"1_18.txt");
            text2_files.push_back(text2location+"1_19.txt");
            text2_files.push_back(text2location+"1_20.txt");
            text2_files.push_back(text2location+"1_21.txt");
            text2_files.push_back(text2location+"1_22.txt");
            text2_files.push_back(text2location+"2_1.txt");
            text2_files.push_back(text2location+"2_2.txt");
            text2_files.push_back(text2location+"2_3.txt");
            text2_files.push_back(text2location+"2_4.txt");
            text2_files.push_back(text2location+"2_5.txt");
            text2_files.push_back(text2location+"2_6.txt");
            text2_files.push_back(text2location+"2_7.txt");
            text2_files.push_back(text2location+"2_8.txt");
            text2_files.push_back(text2location+"2_9.txt");
            text2_files.push_back(text2location+"2_10.txt");
            text2_files.push_back(text2location+"2_11.txt");
            text2_files.push_back(text2location+"2_12.txt");
            text2_files.push_back(text2location+"2_13.txt");
            text2_files.push_back(text2location+"2_14.txt");
            text2_files.push_back(text2location+"2_15.txt");
            text2_files.push_back(text2location+"2_16.txt");
            text2_files.push_back(text2location+"2_17.txt");
            text2_files.push_back(text2location+"2_18.txt");
            text2_files.push_back(text2location+"2_19.txt");
            text2_files.push_back(text2location+"2_20.txt");
            text2_files.push_back(text2location+"2_21.txt");
            text2_files.push_back(text2location+"2_22.txt");
            text2_files.push_back(text2location+"2_23.txt");
            text2_files.push_back(text2location+"3_1.txt");
            text2_files.push_back(text2location+"3_2.txt");
            text2_files.push_back(text2location+"3_3.txt");
            text2_files.push_back(text2location+"3_4.txt");
            text2_files.push_back(text2location+"3_5.txt");
            text2_files.push_back(text2location+"3_6.txt");
            text2_files.push_back(text2location+"3_7.txt");
            text2_files.push_back(text2location+"3_8.txt");
            text2_files.push_back(text2location+"3_9.txt");
            text2_files.push_back(text2location+"3_10.txt");
            text2_files.push_back(text2location+"3_11.txt");
            text2_files.push_back(text2location+"3_12.txt");
            text2_files.push_back(text2location+"3_13.txt");
            text2_files.push_back(text2location+"3_14.txt");
            text2_files.push_back(text2location+"3_15.txt");
            text2_files.push_back(text2location+"4_1.txt");
            text2_files.push_back(text2location+"4_2.txt");
            text2_files.push_back(text2location+"4_3.txt");
            text2_files.push_back(text2location+"4_4.txt");
            text2_files.push_back(text2location+"4_5.txt");
            text2_files.push_back(text2location+"4_6.txt");
            text2_files.push_back(text2location+"4_7.txt");
            text2_files.push_back(text2location+"4_8.txt");
            text2_files.push_back(text2location+"4_9.txt");
            text2_files.push_back(text2location+"4_10.txt");
            text2_files.push_back(text2location+"4_11.txt");
            text2_files.push_back(text2location+"4_12.txt");
            text2_files.push_back(text2location+"4_13.txt");
            text2_files.push_back(text2location+"4_14.txt");
            text2_files.push_back(text2location+"4_15.txt");
            text2_files.push_back(text2location+"4_16.txt");
            text2_files.push_back(text2location+"4_17.txt");
            text2_files.push_back(text2location+"5_1.txt");
            text2_files.push_back(text2location+"5_2.txt");
            text2_files.push_back(text2location+"5_3.txt");
            text2_files.push_back(text2location+"5_4.txt");
            text2_files.push_back(text2location+"5_5.txt");
            text2_files.push_back(text2location+"5_6.txt");
            text2_files.push_back(text2location+"5_7.txt");
            text2_files.push_back(text2location+"5_8.txt");
            text2_files.push_back(text2location+"5_9.txt");
            text2_files.push_back(text2location+"5_10.txt");
            text2_files.push_back(text2location+"5_11.txt");
            text2_files.push_back(text2location+"5_12.txt");
            text2_files.push_back(text2location+"5_13.txt");
            text2_files.push_back(text2location+"5_14.txt");
            text2_files.push_back(text2location+"6_1.txt");
            text2_files.push_back(text2location+"6_2.txt");
            text2_files.push_back(text2location+"6_3.txt");
            text2_files.push_back(text2location+"6_4.txt");
            text2_files.push_back(text2location+"6_5.txt");
            text2_files.push_back(text2location+"6_6.txt");
            text2_files.push_back(text2location+"6_7.txt");
            text2_files.push_back(text2location+"6_8.txt");
            text2_files.push_back(text2location+"6_9.txt");
            text2_files.push_back(text2location+"6_10.txt");
            text2_files.push_back(text2location+"6_11.txt");
            text2_files.push_back(text2location+"6_12.txt");
            text2_files.push_back(text2location+"6_13.txt");
            text2_files.push_back(text2location+"6_14.txt");
            text2_files.push_back(text2location+"7_1.txt");
            text2_files.push_back(text2location+"7_2.txt");
            text2_files.push_back(text2location+"7_3.txt");
            text2_files.push_back(text2location+"7_4.txt");
            text2_files.push_back(text2location+"7_5.txt");
            text2_files.push_back(text2location+"7_6.txt");
            text2_files.push_back(text2location+"7_7.txt");
            text2_files.push_back(text2location+"7_8.txt");
            text2_files.push_back(text2location+"7_9.txt");
            text2_files.push_back(text2location+"7_10.txt");
            text2_files.push_back(text2location+"8_1.txt");
            text2_files.push_back(text2location+"8_2.txt");
            text2_files.push_back(text2location+"8_3.txt");
            text2_files.push_back(text2location+"8_4.txt");
            text2_files.push_back(text2location+"8_5.txt");
            text2_files.push_back(text2location+"8_6.txt");
            text2_files.push_back(text2location+"8_7.txt");
            text2_files.push_back(text2location+"8_8.txt");
            text2_files.push_back(text2location+"8_9.txt");
            text2_files.push_back(text2location+"8_10.txt");
            text2_files.push_back(text2location+"8_11.txt");
            text2_files.push_back(text2location+"8_12.txt");
            text2_files.push_back(text2location+"8_13.txt");
            text2_files.push_back(text2location+"8_14.txt");
            text2_files.push_back(text2location+"8_15.txt");
            text2_files.push_back(text2location+"8_16.txt");
            text2_files.push_back(text2location+"8_17.txt");
            text2_files.push_back(text2location+"9_1.txt");
            text2_files.push_back(text2location+"9_2.txt");
            text2_files.push_back(text2location+"9_3.txt");
            text2_files.push_back(text2location+"9_4.txt");
            text2_files.push_back(text2location+"9_5.txt");
            text2_files.push_back(text2location+"9_6.txt");
            text2_files.push_back(text2location+"9_7.txt");
            text2_files.push_back(text2location+"9_8.txt");
            text2_files.push_back(text2location+"9_9.txt");
            text2_files.push_back(text2location+"9_10.txt");
            text2_files.push_back(text2location+"9_11.txt");
            text2_files.push_back(text2location+"9_12.txt");
            text2_files.push_back(text2location+"9_13.txt");
            text2_files.push_back(text2location+"9_14.txt");
            text2_files.push_back(text2location+"9_15.txt");
            text2_files.push_back(text2location+"9_16.txt");
            text2_files.push_back(text2location+"9_17.txt");
            text2_files.push_back(text2location+"9_18.txt");
            text2_files.push_back(text2location+"9_19.txt");
            text2_files.push_back(text2location+"9_20.txt");
            text2_files.push_back(text2location+"9_21.txt");
            text2_files.push_back(text2location+"9_22.txt");
            text2_files.push_back(text2location+"9_23.txt");
            text2_files.push_back(text2location+"9_24.txt");
            text2_files.push_back(text2location+"9_25.txt");
            text2_files.push_back(text2location+"9_26.txt");
            text2_files.push_back(text2location+"9_27.txt");
            text2_files.push_back(text2location+"9_28.txt");
            text2_files.push_back(text2location+"9_29.txt");
            text2_files.push_back(text2location+"9_30.txt");
            text2_files.push_back(text2location+"9_31.txt");
            text2_files.push_back(text2location+"9_32.txt");
            text2_files.push_back(text2location+"10_1.txt");
            text2_files.push_back(text2location+"10_2.txt");
            text2_files.push_back(text2location+"10_3.txt");
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

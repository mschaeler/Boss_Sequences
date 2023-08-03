//
// Created by Martin on 17.07.2023.
//

#ifndef PRANAY_TEST_PERMUTATIONSOLVER_H
#define PRANAY_TEST_PERMUTATIONSOLVER_H

#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

class PermutationSolver{
private:
    const int k;
    const vector<vector<int>> permutations;

    vector<vector<int>> permute(const int k) {
        vector<int> temp(k);
        for(int i=0;i<k;i++){
            temp.at(i) = i;
        }
        return permute(temp);
    }

    vector<vector<int>> permute(const vector<int>& nums) {
        vector<vector<int>> results;
        vector<int> permutation(nums);

        do {
            results.push_back(permutation);
        } while (std::next_permutation(permutation.begin(), permutation.end()));

        return results;
    }

public:
    PermutationSolver(int _k) : k(_k), permutations(permute(_k)){
        //permutations = permute(_k);
    }

    double solve(const vector<vector<double>>& cost_matrix) const {
        double min_distance = 10000;//some big value
        for(vector<int> perm : permutations){
            double distance = 0;
            for(int mapping=0;mapping<k;mapping++){
                const int mapped_to = perm.at(mapping);
                const double mapping_sim = cost_matrix.at(mapping).at(mapped_to);
                distance += mapping_sim;
            }
            if(distance < min_distance){
                min_distance = distance;
            }
        }

        return min_distance;//return normalized similarity
    }

    double solve(const vector<vector<double>>& cost_matrix, vector<int>& assignment) const {
        double max_similarity = 0;
        for(vector<int> perm : permutations){
            double similarity = 0;
            for(int mapping=0;mapping<k;mapping++){
                const int mapped_to = perm.at(mapping);
                const double mapping_sim = cost_matrix.at(mapping).at(mapped_to);
                similarity += mapping_sim;
            }
            if(similarity>max_similarity){
                max_similarity = similarity;
                assignment = perm;
            }
        }

        return max_similarity;
    }

    void out(){
        for(auto perm : permutations){
            for(auto id : perm){
                cout << id << " ";
            }
            cout <<endl;
        }
    }
};
#endif //PRANAY_TEST_PERMUTATIONSOLVER_H

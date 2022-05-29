//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 22/5/22.
//

#ifndef utils_h
#define utils_h

#include <random>
#include <fstream>

// Utility class to generate random numbers.
class NumberGenerator {
private:
    std::mt19937 seed;
    std::uniform_int_distribution<> generator;
    
public:
    NumberGenerator(int min, int max) {
        std::random_device rd;
        seed = std::mt19937(rd());
        generator = std::uniform_int_distribution<>(min, max);
    }
    
    int genRandomNumber() {
        return generator(seed);
    }
};

// Utility function to read the string content of a file.
std::string readFromFile(const std::string filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        auto start = std::istreambuf_iterator<char>(file);
        auto end = std::istreambuf_iterator<char>();
        return std::string(start, end);
    }
    else {
        std::cout << "File '" << filename << "' not found." << std::endl;
        std::exit(1);
    }
}

// Utility function to write string content to a file.
void writeToFile(const std::string filename, const std::string content) {
    std::ofstream file;
    file.open(filename);
    file << content;
    file.close();
}

// Utility function to find the index of an element in a given array.
template <typename T>
int findIndex(T arr[], size_t len, T target) {
    // An alternative way is to just use std::distance with std::find but this is simpler.
    for (int i = 0; i < len; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

#endif

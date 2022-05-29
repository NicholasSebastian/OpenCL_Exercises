//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 24/5/22.
//

#ifndef cipher_h
#define cipher_h

#include <functional>
#include "main.h"

#define LOWERCASE_OFFSET 97
#define UPPERCASE_OFFSET 65
#define ALPHABET_COUNT 26

// Task 2a
class ShiftCipher {
private:
    int stride;

    std::string process(const std::string input, const std::function<char(char, int)> shift) {
        std::string output = "";
        for (int i = 0; i < input.length(); i++) {
            char ch = input[i];
            int offset = std::isupper(ch) ? UPPERCASE_OFFSET : LOWERCASE_OFFSET;
            output += std::isalpha(ch) ? shift(ch, offset) : ch;
        }
        return output;
    }
    
public:
    ShiftCipher(int n): stride(n) {}
    
    std::string encrypt(const std::string input) {
        return process(input, [&](char ch, int offset) {
            return ((ch + stride - offset) % ALPHABET_COUNT) + offset;
        });
    }
    
    std::string decrypt(const std::string input) {
        return process(input, [&](char ch, int offset) {
            return (ch - offset < stride) ? (ALPHABET_COUNT - (stride - ch)) : (ch - stride);
        });
    }
};

// Base class to help with Task 2b and 2c.
class StringProcessor {
protected:
    Processor* processor;
    cl::CommandQueue* queue;
    
    std::string process(const std::string input, cl::Kernel* kernel) {
        size_t len = input.length();
        
        // Create a copy of the input string as an char array.
        char string[len];
        input.copy(string, len);
        
        // Create memory buffers to pass as the kernel arguments.
        cl::Buffer* input_d = processor->createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, len, &string[0]);
        cl::Buffer* output_d = processor->createBuffer(CL_MEM_WRITE_ONLY, len);
        
        // Define the kernel call.
        kernel->setArg(0, *input_d);
        kernel->setArg(1, *output_d);
        
        // Enqueue/execute the kernel call.
        queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(len));
        queue->enqueueReadBuffer(*output_d, CL_TRUE, 0, len, &string[0]);
        
        delete input_d;
        delete output_d;
        return std::string(string);
    }
    
public:
    StringProcessor(Processor* p, std::string filename): processor(p) {
        p->compileAndUse(filename);
        queue = p->getQueue();
    }
};

// Task 2b
class ParallelShiftCipher : StringProcessor {
private:
    int stride;
    
    std::string shift(const std::string input, std::string kernelname) {
        cl::Kernel* task = processor->createTask(kernelname);
        task->setArg(2, stride);
        std::string output = process(input, task);
        delete task;
        return output;
    }
    
public:
    ParallelShiftCipher(Processor* p, int n): StringProcessor(p, "task2.cl"), stride(n) {}
    
    std::string encrypt(const std::string input) {
        return shift(input, "encrypt");
    }
    
    std::string decrypt(const std::string input) {
        return shift(input, "decrypt");
    }
};

// Task 2c
class ParallelSubstituteCipher : StringProcessor {
private:
    cl::Buffer* lookupTable_d;
    cl::Buffer* reverseTable_d;
    
    std::string substitute(const std::string input, cl::Buffer* lookup) {
        cl::Kernel* task = processor->createTask("substitute");
        task->setArg(2, *lookup);
        std::string output = process(input, task);
        delete task;
        return output;
    }
    
    // O(n²) complexity. This overhead can be reduced by also turning this into a kernel function.
    // For our use case, since n is only 26, and will only be calculated once throughout the lifetime of the instance,
    // leaving it this way is fine.
    static char* genReverseLookup(char* lookup) {
        char* reverse = new char[ALPHABET_COUNT];
        for (int i = 0; i < ALPHABET_COUNT; i++) {
            char ch = UPPERCASE_OFFSET + i;
            int chIndex = findIndex(lookup, ALPHABET_COUNT, ch);
            reverse[i] = UPPERCASE_OFFSET + chIndex;
        }
        return reverse;
    }
    
public:
    ParallelSubstituteCipher(Processor* p, char* lookupTable): StringProcessor(p, "task2.cl") {
        char* reverseTable = genReverseLookup(lookupTable);
        
        lookupTable_d = processor->createBuffer(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, ALPHABET_COUNT, lookupTable);
        reverseTable_d = processor->createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ALPHABET_COUNT, reverseTable);
        
        delete[] reverseTable;
    }
    
    ~ParallelSubstituteCipher() {
        delete lookupTable_d;
        delete reverseTable_d;
    }
    
    std::string encrypt(const std::string input) {
        return substitute(input, lookupTable_d);
    }
    
    std::string decrypt(const std::string input) {
        return substitute(input, reverseTable_d);
    }
};

#endif

//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 23/5/22.
//

#define LOWERCASE_OFFSET 97
#define UPPERCASE_OFFSET 65
#define ALPHABET_COUNT 26

bool isupper(char ch) {
    return ch >= 'A' && ch <= 'Z';
}

bool islower(char ch) {
    return ch >= 'a' && ch <= 'z';
}

bool isalpha(char ch) {
    return isupper(ch) || islower(ch);
}

__kernel void encrypt(__global const char* input, __global char* output, const int stride) {
    int i = get_global_id(0);
    char ch = input[i];
    if (isalpha(ch)) {
        int offset = isupper(ch) ? UPPERCASE_OFFSET : LOWERCASE_OFFSET;
        output[i] = ((ch + stride - offset) % ALPHABET_COUNT) + offset;
    }
    else output[i] = ch;
}

__kernel void decrypt(__global const char* input, __global char* output, const int stride) {
    int i = get_global_id(0);
    char ch = input[i];
    if (isalpha(ch)) {
        int offset = isupper(ch) ? UPPERCASE_OFFSET : LOWERCASE_OFFSET;
        output[i] = (ch - offset < stride) ? (ALPHABET_COUNT - (stride - ch)) : (ch - stride);
    }
    else output[i] = ch;
}

__kernel void substitute(__global const char* input, __global char* output, __global const char* lookup) {
    int i = get_global_id(0);
    char ch = input[i];
    if (isupper(ch))
        output[i] = lookup[ch - UPPERCASE_OFFSET];
    else if (islower(ch))
        output[i] = lookup[ch - LOWERCASE_OFFSET] + (LOWERCASE_OFFSET - UPPERCASE_OFFSET);
    else
        output[i] = ch;
}

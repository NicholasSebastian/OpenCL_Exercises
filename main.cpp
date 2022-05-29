//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 22/5/22.
//

#include "image.h"

using namespace std;
using namespace cl;

vector<Platform> platforms;
vector<Device> devices;
Processor* processor;

void task1() {
    NumberGenerator rand(10, 20);
    int vec1[8];
    int vec2[16];
    int out[32];
    
    // Initialize the input array values.
    for (int i = 0; i < 8; i++)
        vec1[i] = rand.genRandomNumber();
    for (int i = 0; i < 16; i++)
        vec2[i] = (i < 8) ? (i + 1) : (-17 + i);
    
    // Read and compile the kernel program.
    processor->compileAndUse("task1.cl");
    
    // Create memory buffers to pass as the kernel arguments.
    Buffer* vec1_d = processor->createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(vec1), &vec1[0]);
    Buffer* vec2_d = processor->createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(vec2), &vec2[0]);
    Buffer* out_d = processor->createBuffer(CL_MEM_READ_ONLY, sizeof(out));
    
    // Define the kernel call.
    Kernel* task = processor->createTask("godHelpMe");
    task->setArg(0, *vec1_d);
    task->setArg(1, *vec2_d);
    task->setArg(2, *out_d);
    
    // Enqueue/execute the kernel call.
    CommandQueue* queue = processor->getQueue();
    queue->enqueueNDRangeKernel(*task, NullRange, NDRange(4));
    queue->enqueueReadBuffer(*out_d, CL_TRUE, 0, sizeof(out), &out[0]);
    
    // Print the content of the output array.
    cout << "\nOutput:" << endl;
    for (int el : out) cout << el << ' ';
    cout << endl;
    
    delete vec1_d;
    delete vec2_d;
    delete out_d;
    delete task;
}

void task2a() {
    // Get the user input.
    int n;
    cout << "Enter an 'n' value: ";
    cin >> n;
    cout << endl;
    
    // Read the text content of the input file.
    string plaintext = readFromFile("plaintext.txt");
    
    // Encrypt the text content, then write it to a file.
    ShiftCipher cipher(n);
    string ciphertext = cipher.encrypt(plaintext);
    writeToFile("ciphertext.txt", ciphertext);
    
    // Decrypt the cipher text, then write it to a file.
    string decrypted = cipher.decrypt(ciphertext);
    writeToFile("decrypted.txt", decrypted);
    
    cout << "Caesar's Cipher process completed." << endl;
}

void task2b() {
    // Get the user input.
    int n;
    cout << "Enter an 'n' value: ";
    cin >> n;
    cout << endl;
    
    // Read the text content of the input file.
    string plaintext = readFromFile("plaintext.txt");
    
    // Encrypt the text content, then write it to a file.
    ParallelShiftCipher cipher(processor, n);
    string ciphertext = cipher.encrypt(plaintext);
    writeToFile("ciphertext.txt", ciphertext);
    
    // Decrypt the cipher text, then write it to a file.
    string decrypted = cipher.decrypt(ciphertext);
    writeToFile("decrypted.txt", decrypted);
    
    cout << "Caesar's Cipher process completed." << endl;
}

void task2c() {
    // Define the lookup table.
    char* lookupTable = new char[ALPHABET_COUNT] {
        'G', 'X', 'S', 'Q', 'F', 'A', 'R', 'O', 'W', 'B', 'L', 'M', 'T',
        'H', 'C', 'V', 'P', 'N', 'Z', 'U', 'I', 'E', 'Y', 'D', 'K', 'J'
    };
    
    // Read the text content of the input file.
    string plaintext = readFromFile("plaintext.txt");
    
    // Encrypt the text content, then write it to a file.
    ParallelSubstituteCipher cipher(processor, lookupTable);
    string ciphertext = cipher.encrypt(plaintext);
    writeToFile("ciphertext.txt", ciphertext);
    
    // Decrypt the cipher text, then write it to a file.
    string decrypted = cipher.decrypt(ciphertext);
    writeToFile("decrypted.txt", decrypted);
    
    cout << "Caesar's Cipher process completed." << endl;
    delete[] lookupTable;
}

void task3a() {
    // Image Luminance.
    ImageManipulator manipulator(processor);
    manipulator.genLuminance("peppers.bmp", "luminance.bmp");
}

void task3bi() {
    // Gaussian Blurring with one-pass.
    ImageManipulator manipulator(processor);
    manipulator.genBlurring("peppers.bmp", "blur.bmp");
}

void task3bii() {
    // Gaussian Blurring with two-passes.
    ImageManipulator manipulator(processor);
    manipulator.genBlurring2("peppers.bmp", "blur2.bmp");
}

void task3c() {
    // Get the user input.
    float threshold;
    cout << "Enter the luminance threshold: ";
    cin >> threshold;
    cout << endl;
    
    // Bloom Effect.
    ImageManipulator manipulator(processor);
    manipulator.genBloom("peppers.bmp", "bloom.bmp", threshold);
}

int main() {
    Platform::get(&platforms);
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    processor = new Processor(&devices.front());
    
    string option;
    cout << "\nSelect a Task to run (1/2a/2b/2c/3a/3bi/3bii/3c): ";
    cin >> option;
    cout << endl;
    
    if (option == "1") task1();
    else if (option == "2a") task2a();
    else if (option == "2b") task2b();
    else if (option == "2c") task2c();
    else if (option == "3a") task3a();
    else if (option == "3bi") task3bi();
    else if (option == "3bii") task3bii();
    else if (option == "3c") task3c();
    else cout << "Invalid input." << endl;
    
    delete processor;
    return 0;
}

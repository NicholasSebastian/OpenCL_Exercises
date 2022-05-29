//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 24/5/22.
//

#ifndef image_h
#define image_h

#include "cipher.h"
#include "bmp_utils.h"

class ImageManipulator {
private:
    Processor* processor;
    cl::CommandQueue* queue;
    cl::ImageFormat imageFormat;
    
    cl::Image2D* execute(cl::Kernel* kernel, int imageWidth, int imageHeight) {
        // Create memory buffer to pass as the kernel output argument.
        cl::Image2D* outputBuffer = processor->createImageBuffer(CL_MEM_READ_WRITE, imageFormat, imageWidth, imageHeight);
        kernel->setArg(0, *outputBuffer);
        
        // Enqueue/execute the kernel call.
        queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(imageWidth, imageHeight));
        return outputBuffer;
    }
    
    unsigned char* bufferToArray(cl::Image2D* buffer, int imageWidth, int imageHeight) {
        // Allocate memory for the array.
        int imageSize = imageWidth * imageHeight * 4;
        unsigned char* output = new unsigned char[imageSize];
        
        // Arguments to read the image out of the buffer later.
        // Initializer lists don't work for cl::size_t? This looks horrendous.
        cl::size_t<3> origin, region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = imageWidth;
        region[1] = imageHeight;
        region[2] = 1;
        
        // Read the data from the buffer onto the array.
        queue->enqueueReadImage(*buffer, CL_TRUE, origin, region, 0, 0, &output[0]);
        return output;
    }
    
public:
    ImageManipulator(Processor* p): processor(p) {
        p->compileAndUse("task3.cl");
        queue = p->getQueue();
        imageFormat = cl::ImageFormat(CL_RGBA, CL_UNORM_INT8);
    }
    
    // Task 3a: Image Luminance.
    void genLuminance(const std::string inputFilename, const std::string outputFilename) {
        int imageWidth, imageHeight;
        
        // Read the given image file onto an input buffer.
        auto input = read_BMP_RGB_to_RGBA(inputFilename.c_str(), &imageWidth, &imageHeight);
        auto inputBuffer = processor->createImageBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, &input[0]);
        
        // Process the input buffer through the kernel function.
        cl::Kernel* task = processor->createTask("luminance");
        task->setArg(1, *inputBuffer);
        auto outputBuffer = execute(task, imageWidth, imageHeight);
        
        // Write the output as an image file.
        auto output = bufferToArray(outputBuffer, imageWidth, imageHeight);
        write_BMP_RGBA_to_RGB(outputFilename.c_str(), output, imageWidth, imageHeight);
        
        // Free memory.
        delete task;
        delete inputBuffer;
        delete outputBuffer;
        delete[] input;
        delete[] output;
    }
    
    // Task 3b i: One-pass Gaussian Blurring.
    void genBlurring(const std::string inputFilename, const std::string outputFilename) {
        int imageWidth, imageHeight;
        
        // Read the given image file onto an input buffer.
        auto input = read_BMP_RGB_to_RGBA(inputFilename.c_str(), &imageWidth, &imageHeight);
        auto inputBuffer = processor->createImageBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, &input[0]);
        
        // Process the input buffer through the kernel function.
        cl::Kernel* task = processor->createTask("gaussian");
        task->setArg(1, *inputBuffer);
        auto outputBuffer = execute(task, imageWidth, imageHeight);
        
        // Write the output as an image file.
        auto output = bufferToArray(outputBuffer, imageWidth, imageHeight);
        write_BMP_RGBA_to_RGB(outputFilename.c_str(), output, imageWidth, imageHeight);
        
        // Free memory.
        delete task;
        delete inputBuffer;
        delete outputBuffer;
        delete[] input;
        delete[] output;
    }
    
    // Task 3b ii: Two-pass Gaussian Blurring.
    void genBlurring2(const std::string inputFilename, const std::string outputFilename) {
        int imageWidth, imageHeight;
        
        // Read the given image file onto an input buffer.
        auto input = read_BMP_RGB_to_RGBA(inputFilename.c_str(), &imageWidth, &imageHeight);
        auto inputBuffer = processor->createImageBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, &input[0]);
        
        // First pass: Horizontal Blurring.
        cl::Kernel* task1 = processor->createTask("gaussianHorizontal");
        task1->setArg(1, *inputBuffer);
        auto outputBuffer1 = execute(task1, imageWidth, imageHeight);
        delete task1;
        
        // Second pass: Vertical Blurring.
        cl::Kernel* task2 = processor->createTask("gaussianVertical");
        task2->setArg(1, *outputBuffer1);
        auto outputBuffer2 = execute(task2, imageWidth, imageHeight);
        delete task2;
        
        // Write the output as an image file.
        auto output = bufferToArray(outputBuffer2, imageWidth, imageHeight);
        write_BMP_RGBA_to_RGB(outputFilename.c_str(), output, imageWidth, imageHeight);
        
        // Free memory.
        delete inputBuffer;
        delete outputBuffer1;
        delete outputBuffer2;
        delete[] input;
        delete[] output;
    }
    
    // Task 3c: Bloom Effect.
    void genBloom(const std::string inputFilename, const std::string outputFilename, float threshold) {
        int imageWidth, imageHeight;
        
        // Read the given image file onto an input buffer.
        auto input = read_BMP_RGB_to_RGBA(inputFilename.c_str(), &imageWidth, &imageHeight);
        auto inputBuffer = processor->createImageBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, &input[0]);
        
        // Filter out the pixels by luminance threshold.
        cl::Kernel* task1 = processor->createTask("luminanceFilter");
        task1->setArg(1, *inputBuffer);
        task1->setArg(2, threshold);
        auto outputBuffer1 = execute(task1, imageWidth, imageHeight);
        delete task1;
        
        // Blur the image horizontally.
        cl::Kernel* task2 = processor->createTask("gaussianHorizontal");
        task2->setArg(1, *outputBuffer1);
        auto outputBuffer2 = execute(task2, imageWidth, imageHeight);
        delete task2;
        
        // Blur the image horizontally.
        cl::Kernel* task3 = processor->createTask("gaussianVertical");
        task3->setArg(1, *outputBuffer2);
        auto outputBuffer3 = execute(task3, imageWidth, imageHeight);
        delete task3;
        
        // Add together the pixel values of the original image and output2's image.
        cl::Kernel* task4 = processor->createTask("merge");
        task4->setArg(1, *inputBuffer);
        task4->setArg(2, *outputBuffer3);
        auto outputBuffer4 = execute(task4, imageWidth, imageHeight);
        delete task4;
        
        // Convert the output buffers into arrays.
        auto output1 = bufferToArray(outputBuffer1, imageWidth, imageHeight);
        auto output2 = bufferToArray(outputBuffer2, imageWidth, imageHeight);
        auto output3 = bufferToArray(outputBuffer3, imageWidth, imageHeight);
        auto output4 = bufferToArray(outputBuffer4, imageWidth, imageHeight);
        
        // Output the data arrays as image files.
        write_BMP_RGBA_to_RGB(("glow_" + outputFilename).c_str(), output1, imageWidth, imageHeight);
        write_BMP_RGBA_to_RGB(("blurH_" + outputFilename).c_str(), output2, imageWidth, imageHeight);
        write_BMP_RGBA_to_RGB(("blur2_" + outputFilename).c_str(), output3, imageWidth, imageHeight);
        write_BMP_RGBA_to_RGB(("final_" + outputFilename).c_str(), output4, imageWidth, imageHeight);
        
        // Free memory.
        delete[] input;
        delete[] output1;
        delete[] output2;
        delete[] output3;
        delete[] output4;
        delete inputBuffer;
        delete outputBuffer1;
        delete outputBuffer2;
        delete outputBuffer3;
        delete outputBuffer4;
    }
};

#endif

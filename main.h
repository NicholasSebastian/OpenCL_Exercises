//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 22/5/22.
//

#ifndef main_h
#define main_h

#include <iostream>
#include <vector>
#include "utils.h"

#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

class Processor {
private:
    cl::Device* device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    
public:
    Processor(cl::Device* device) {
        this->device = device;
        this->context = cl::Context(*device);
        this->queue = cl::CommandQueue(context, *device);
        std::cout << "Using the device: " << device->getInfo<CL_DEVICE_NAME>() << std::endl;
    }
    
    void compileAndUse(const std::string filename) {
        std::string sourceContent = readFromFile(filename);
        cl::Program::Sources sources(1, std::make_pair(sourceContent.c_str(), sourceContent.length() + 1));
        this->program = cl::Program(context, sources);
        try {
            program.build();
            std::cout << "Program built successfully." << std::endl;
        }
        catch (const std::exception& e) {
            std::cout << "An error occurred when building the program." << std::endl;
            std::exit(1);
        }
    }
    
    cl::CommandQueue* getQueue() {
        return &queue;
    }
    
    cl::Kernel* createTask(const std::string kernelname) {
        return new cl::Kernel(program, kernelname.c_str());
    }
    
    cl::Buffer* createBuffer(cl_mem_flags flag, size_t size) {
        return new cl::Buffer(context, flag, size);
    }
    
    cl::Buffer* createBuffer(cl_mem_flags flag, size_t size, void* data) {
        return new cl::Buffer(context, flag, size, data);
    }
    
    cl::Image2D* createImageBuffer(cl_mem_flags flag, cl::ImageFormat &fmt, int w, int h) {
        return new cl::Image2D(context, flag, fmt, w, h);
    }
    
    cl::Image2D* createImageBuffer(cl_mem_flags flag, cl::ImageFormat &fmt, int w, int h, void* data) {
        return new cl::Image2D(context, flag, fmt, w, h, 0, data);
    }
};

#endif

//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 29/5/22.
//  Contents copied from "bmpfuncs.cpp" of CSCI376 Tutorial 7.
//

#ifndef bmp_utils_h
#define bmp_utils_h

// Reads the contents of a 24-bit RGB bitmap file and returns it in RGBA format.
unsigned char* read_BMP_RGB_to_RGBA(const char *filename, int* widthOut, int* heightOut)
{
    char fileHeader[54];    // to store the file header, bmp file format bmpheader (14 bytes) + bmpheaderinfo (40 bytes) = 54 bytes
    int width, height;      // width and height of image
    int offset;             // offset where image data starts in the file
    int imageSize;          // image size in bytes
    int padSize;            // in the bmp file format, each row must be a multiple of 4
    int rowSize;            // size per row in bytes
    int i, j;
    char colour[3];
    unsigned char* imageData;    // pointer to store image data

    // open file stream
    std::ifstream textureFileStream(filename, std::ios::in | std::ios::binary);

    // check whether file opened successfully
    if (!textureFileStream.is_open())
    {
        std::cout << "Failed to open texture file - " << filename << std::endl;
        return NULL;
    }

    // get file header
    textureFileStream.read(fileHeader, 54);

    // get offset, width and height information
    offset = fileHeader[10];
    width = fileHeader[21] << 24;
    width |= fileHeader[20] << 16;
    width |= fileHeader[19] << 8;
    width |= fileHeader[18];
    width = abs(width);
    height = fileHeader[25] << 24;
    height |= fileHeader[24] << 16;
    height |= fileHeader[23] << 8;
    height |= fileHeader[22];
    height = abs(height);

    // allocate RGBA image data
    imageSize = width * height * 4;
    imageData = new unsigned char[imageSize];

    // move read position by offset
    textureFileStream.seekg(offset, std::ios::beg);

    // compute amount of row padding
    padSize = width % 4;
    if (padSize != 0) {
        padSize = 4 - padSize;
    }

    // bitmaps are stored in upside-down raster order
    rowSize = width * 4;

    // read each RGB pixel colour
    for (i = 0; i < height; i++) {
        for (j = 0; j < rowSize; j += 4) {
            textureFileStream.read(colour, 3);
            imageData[i*rowSize + j] = colour[0];
            imageData[i*rowSize + j + 1] = colour[1];
            imageData[i*rowSize + j + 2] = colour[2];
            imageData[i*rowSize + j + 3] = 255;
        }
        // in the bmp format, each row has to be a multiple of 4
        // read in the junk data and throw it away
        for (j = 0; j < padSize; j++) {
            textureFileStream.read(colour, 3);
        }
    }

    // close file stream
    textureFileStream.close();

    // record width and height, and return pointer to image data
    *widthOut = width;
    *heightOut = height;

    return imageData;
}

// Writes imageData (in RGBA format) to a 24-bit RGB bitmap file.
void write_BMP_RGBA_to_RGB(const char *filename, unsigned char* imageData, int width, int height)
{
    char fileHeader[54] = {
        // BITMAPHEADER
        'B','M',        // bmp file
        0, 0, 0, 0,        // file size in bytes
        0, 0,            // reserved
        0, 0,            // reserved
        54, 0, 0, 0,    // offset
        // BITMAPINFOHEADER
        40, 0, 0, 0,    // size of info header
        0, 0, 0, 0,        // width
        0, 0, 0, 0,        // heigth
        1, 0,            // number colour planes
        24, 0,            // number of bits per pixel
        0, 0, 0, 0,        // compression
        0, 0, 0, 0,        // image size
        0, 0, 0, 0,        // horizontal resolution
        0, 0, 0, 0,        // vertical resolution
        0, 0, 0, 0,        // number of colours in palette
        0, 0, 0, 0,        // number of important colours
    };
    int imageSize;        // image size in bytes
    int padSize;        // in the bmp file format, each row must be a multiple of 4
    int rowSize;        // size per row in bytes
    int fileSize;        // file size in bytes (image size + header size)
    int i, j;
    char colour[3];

    // compute amount of padding so that each row is a multiple of 4
    padSize = width % 4;
    if (padSize != 0) {
        padSize = 4 - padSize;
    }

    // compute image size
    imageSize = (width + padSize) * height * 3;

    // open output stream
    std::ofstream outFileStream(filename, std::ios::out | std::ios::binary);

    // check whether output stream opened successfully
    if (!outFileStream.is_open())
    {
        std::cout << "Failed to open output file - " << filename << std::endl;
        return;
    }

    // compute file size (image size + header size)
    fileSize = 54 + imageSize;

    // fill in appropriate bmp header fields in little endian order
    fileHeader[2] = (unsigned char)fileSize;
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    fileHeader[18] = (unsigned char)width;
    fileHeader[19] = (unsigned char)(width >> 8);
    fileHeader[20] = (unsigned char)(width >> 16);
    fileHeader[21] = (unsigned char)(width >> 24);

    fileHeader[22] = (unsigned char)height;
    fileHeader[23] = (unsigned char)(height >> 8);
    fileHeader[24] = (unsigned char)(height >> 16);
    fileHeader[25] = (unsigned char)(height >> 24);

    fileHeader[34] = (unsigned char)imageSize;
    fileHeader[35] = (unsigned char)(imageSize >> 8);
    fileHeader[36] = (unsigned char)(imageSize >> 16);
    fileHeader[37] = (unsigned char)(imageSize >> 24);

    // write file header to out stream
    outFileStream.write(fileHeader, 54);

    // bitmaps are stored in upside-down raster order
    rowSize = width * 4;

    // output each RGB pixel colour
    for (i = 0; i < height; i++) {
        for (j = 0; j < rowSize; j += 4) {
            colour[0] = (unsigned char)imageData[i*rowSize + j];
            colour[1] = (unsigned char)imageData[i*rowSize + j + 1];
            colour[2] = (unsigned char)imageData[i*rowSize + j + 2];
            outFileStream.write(colour, 3);
        }

        // in bmp format rows must be a multiple of 4-bytes, add junk padding if required
        for (j = 0; j < padSize; j++) {
            outFileStream.write(colour, 3);
        }
    }

    // close output file stream
    outFileStream.close();
}

#endif

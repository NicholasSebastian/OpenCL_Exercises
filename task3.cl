//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 24/5/22.
//

// LUMINANCE

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void luminance(write_only image2d_t output, read_only image2d_t input) {
    int2 coordinate = (int2) (get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef(input, sampler, coordinate);

    // Calculate the luminance for each pixel.
    float luminance = (0.299f * pixel.z) + (0.587f * pixel.y) + (0.114f * pixel.x);

    // Set the RGB values of each pixel to the luminance value.
    pixel.xyz = luminance;
    pixel.w = 255;

    write_imagef(output, coordinate, pixel);
}

// GAUSSIAN BLUR

#define WINDOW_WIDTH 7
#define WINDOW_AREA 49
#define WINDOW_RADIUS 3

__constant float weights[WINDOW_AREA] = {
    0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036,
    0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
    0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
    0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291,
    0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446,
    0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363,
    0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036
};

__kernel void gaussian(write_only image2d_t output, read_only image2d_t input) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int filterIndex = 0;
    float4 value = 0.0;
    
    // For every pixel within the 7x7 window around the current coordinate:
    for (int offsetY = -WINDOW_RADIUS; offsetY <= WINDOW_RADIUS; offsetY++) {
        for (int offsetX = -WINDOW_RADIUS; offsetX <= WINDOW_RADIUS; offsetX++) {
            
            // Get the pixel value.
            int2 neighbour = (int2) (x + offsetX, y + offsetY);
            float4 pixel = read_imagef(input, sampler, neighbour);

            // Acculumate the weighted sum.
            value.xyz += pixel.xyz * weights[filterIndex];
            filterIndex++;
        }
    }

    int2 coordinate = (int2) (x, y);
    write_imagef(output, coordinate, value);
}

__constant float flatWeights[WINDOW_WIDTH] = {
    0.00598, 0.060626, 0.241843, 0.383103, 0.241843, 0.060626, 0.00598
};

__kernel void gaussianHorizontal(write_only image2d_t output, read_only image2d_t input) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int filterIndex = 0;
    float4 value = 0.0;
    
    // For every pixel within the 7x1 window horizontal to the current coordinate:
    for (int offsetX = -WINDOW_RADIUS; offsetX <= WINDOW_RADIUS; offsetX++) {
        
        // Get the pixel value.
        int2 neighbour = (int2) (x + offsetX, y);
        float4 pixel = read_imagef(input, sampler, neighbour);
        
        // Acculumate the weighted sum.
        value.xyz += pixel.xyz * flatWeights[filterIndex];
        filterIndex++;
    }
    
    int2 coordinate = (int2) (x, y);
    write_imagef(output, coordinate, value);
}

__kernel void gaussianVertical(write_only image2d_t output, read_only image2d_t input) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    int filterIndex = 0;
    float4 value = 0.0;
    
    // For every pixel within the 1x7 window vertical to the current coordinate:
    for (int offsetY = -WINDOW_RADIUS; offsetY <= WINDOW_RADIUS; offsetY++) {
        
        // Get the pixel value.
        int2 neighbour = (int2) (x, y + offsetY);
        float4 pixel = read_imagef(input, sampler, neighbour);
        
        // Acculumate the weighted sum.
        value.xyz += pixel.xyz * flatWeights[filterIndex];
        filterIndex++;
    }
    
    int2 coordinate = (int2) (x, y);
    write_imagef(output, coordinate, value);
}

// BLOOM EFFECT

__kernel void luminanceFilter(write_only image2d_t output, read_only image2d_t input, float threshold) {
    int2 coordinate = (int2) (get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef(input, sampler, coordinate);

    // Calculate the luminance for each pixel.
    float luminance = (0.299f * pixel.z) + (0.587f * pixel.y) + (0.114f * pixel.x);

    // Pixels below the given luminance threshold are set to black.
    if (luminance < threshold) pixel.xyz = 0;

    write_imagef(output, coordinate, pixel);
}

__kernel void merge(write_only image2d_t output, read_only image2d_t input1, read_only image2d_t input2) {
    int2 coordinate = (int2) (get_global_id(0), get_global_id(1));
    float4 pixel1 = read_imagef(input1, sampler, coordinate);
    float4 pixel2 = read_imagef(input2, sampler, coordinate);
    
    // Literally just add the two values together.
    float4 pixelMix = pixel1 + pixel2;
    
    // Clamp the maximum color value.
    float4 pixelFinal = clamp(pixelMix, 0.0f, 1.0f);

    write_imagef(output, coordinate, pixelFinal);
}

//
//  CSCI376_Assignment2
//  Created by Nicholas Sebastian Hendrata on 22/5/22.
//

__kernel void godHelpMe(__global const int4 *input1, __global const int *input2, __global int *output) {
    __local int8 v, v1, v2;
    __private int8 results;

    // Copy the contents of input1 into an int8 vector called v.
    v = (int8) (input1[0], input1[1]);
    
    // Copy the contents of input2 into two int8 vectors called v1 and v2.
    v1 = vload8(0, input2);
    v2 = vload8(1, input2);

    // Check whether any of the elements in v are greater than 15.
    if (any(v > 15)) {
        // For elements that are greater than 15, copy the corresponding elements from v1 into results.
        // For elements less than or equal to 15, copy the corresponding elements from v2 into results.
        results.s0 = select(v2.s0, v1.s0, isgreater(v.s0, 15));
        results.s1 = select(v2.s1, v1.s1, isgreater(v.s1, 15));
        results.s2 = select(v2.s2, v1.s2, isgreater(v.s2, 15));
        results.s3 = select(v2.s3, v1.s3, isgreater(v.s3, 15));
        results.s4 = select(v2.s4, v1.s4, isgreater(v.s4, 15));
        results.s5 = select(v2.s5, v1.s5, isgreater(v.s5, 15));
        results.s6 = select(v2.s6, v1.s6, isgreater(v.s6, 15));
        results.s7 = select(v2.s7, v1.s7, isgreater(v.s7, 15));
    }
    else {
        // Fill the first 4 elements of results with the contents from the first 4 elements of v1.
        // Fill the next 4 elements of results with contents from the first 4 elements of v2.
        results.s0 = v1.s0;
        results.s1 = v1.s1;
        results.s2 = v1.s2;
        results.s3 = v1.s3;
        results.s4 = v2.s0;
        results.s5 = v2.s1;
        results.s6 = v2.s2;
        results.s7 = v2.s3;
    }

    // Stores the contents of v, v1, v2 and results in the output array.
    vstore8(v, 0, output);
    vstore8(v1, 1, output);
    vstore8(v2, 2, output);
    vstore8(results, 3, output);
}

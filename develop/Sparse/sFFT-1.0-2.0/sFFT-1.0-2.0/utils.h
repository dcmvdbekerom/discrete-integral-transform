/*
 * Copyright (c) 2012-2013 Haitham Hassanieh, Piotr Indyk, Dina Katabi,
 *   Eric Price, Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * 
 */

#ifndef UTILS_H
#define UTILS_H

#include "fft.h"

// Compute the gcd of a and b
// assumes a, b > 0.
int gcd(int a, int b);

// crappy inversion code I stole from elsewhere
// Undefined if gcd(a, n) > 1
int mod_inverse(int a, int n);

/*
  Compute the num'th smallest element of the length n input.
  uses std::nth_element, but doesn't mutate input.
 */
real_t nth_element_immutable(real_t *input, int n, int num);

/*
  Output the indices corresponding to the num largest elements of samples.
  Output is sorted.
*/
void find_largest_indices(int *output, int num, real_t *samples, int n);


void radix(int byte, int size, int *A, int *TEMP);

void radix_sort(int *A, int size);

void radix_filt(int byte, int size, int *A, int *TEMP, complex_t* Filter, complex_t* TMP_F);

void radix_sort_filt(int *A, complex_t* Filter,int size);

int floor_to_pow2(double x);

//x[i] <- x[i-r]
void shift(complex_t *x, int n, int r);

double phase(complex_t x);

double AWGN(complex_t *x, int n, double std_noise);

double cabs2(complex_t x);

double binomial_cdf(double prob, int n, int needed);

#endif

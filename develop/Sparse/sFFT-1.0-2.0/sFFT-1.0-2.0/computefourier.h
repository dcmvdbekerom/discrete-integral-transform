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

#ifndef COMPUTEFOURIER_H
#define COMPUTEFOURIER_H

#include "fft.h"

#include <complex.h>
#include <map>
#include "fftw.h"
#include "filters.h"

#define OPTIMIZE_FFTW 0
//#define  WITH_COMB 0 

extern bool WITH_COMB;
extern bool ALGORITHM1;
extern bool VERBOSE;
extern bool TIMING;

//Comments located in the cc file.
std::map<int, complex_t>
  outer_loop(complex_t *origx, int n, const Filter &filter, const Filter &filter_Est, int B2,
	   int num, int B, int W_Comb, int Comb_loops, int loop_threshold, int location_loops, int loops);

#endif

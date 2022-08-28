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

#include "fftw.h"
#include<map>

std::map<int, fftw_plan> fftw_plans;

int fftw_dft(complex_t *out, int n, complex_t *x, int backwards){
  fftw_plan p;
  if (OPTIMIZE_FFTW) {  //measure the best plan the first time
    if (fftw_plans.find(n) == fftw_plans.end()) { // no plan made yet
      complex_t *in = (complex_t *)fftw_malloc(sizeof(*in)*n);
      complex_t *out2 = (complex_t *)fftw_malloc(sizeof(*out2)*n);
      p = fftw_plan_dft_1d(n, in, out2,
                           backwards? FFTW_BACKWARD:FFTW_FORWARD,
                           FFTW_MEASURE);
      fftw_plans.insert(std::make_pair(n, p));
      fftw_free(in);
      fftw_free(out2);
    }
  }
  p = fftw_plan_dft_1d(n, x, out,
               backwards ? FFTW_BACKWARD:FFTW_FORWARD,
                       FFTW_ESTIMATE);
  fftw_execute(p);
  fftw_destroy_plan(p);
  return 0;
}

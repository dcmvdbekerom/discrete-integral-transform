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

#ifndef PLOT_H
#define PLOT_H

#include "fft.h"
#include <unistd.h>
#include <stdarg.h>
#include <sys/wait.h>
#include <string>
#include<vector>

#define DEFAULT_PREAMBLE "set style data linespoints"

std::vector<real_t> map_abs(std::vector<complex_t> x);
std::vector<real_t> map_real(std::vector<complex_t> x);
std::vector<real_t> map_imag(std::vector<complex_t> x);

int plotn(std::string preamble, std::vector<std::vector<std::pair<real_t, real_t> > > plots, std::string titles="");


int plotn(std::string preamble, std::vector<std::vector<real_t> > plots, std::string titles="");

void plot_fft(std::string preamble, std::vector<complex_t> x, int real=0);


template <typename T>
inline int plot(std::string title, std::string titles, std::vector<T> x,
		std::vector<T> y=std::vector<T>(),
		std::vector<T> z=std::vector<T>(),
		std::vector<T> w=std::vector<T>(),
		std::vector<T> a=std::vector<T>(),
		std::vector<T> b=std::vector<T>()
		){
  std::vector<std::vector<T> > vals;
  vals.push_back(x);
  if (y.size())
    vals.push_back(y);
  if (z.size())
    vals.push_back(z);
  if (w.size())
    vals.push_back(w);
  if (a.size())
    vals.push_back(a);
  if (b.size())
    vals.push_back(b);
  return plotn("set title '" + title + "'", vals, titles);
}

template <typename T>
inline int plot(std::string title, std::vector<T> x,
		std::vector<T> y=std::vector<T>(),
		std::vector<T> z=std::vector<T>(),
		std::vector<T> w=std::vector<T>(),
		std::vector<T> a=std::vector<T>(),
		std::vector<T> b=std::vector<T>()
		){
  return plot(title, "", x, y, z, w, a, b);
}

#endif

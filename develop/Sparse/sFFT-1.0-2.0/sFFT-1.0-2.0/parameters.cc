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

#include "parameters.h"
  
void get_expermient_vs_N_parameters(int N, bool WITH_COMB, double &Bcst_loc, double &Bcst_est,  double &Comb_cst,  int &loc_loops, 
                                      int &est_loops,  int &threshold_loops, int &comb_loops, double &tolerance_loc, double &tolerance_est){

        if(WITH_COMB){

				 switch(N){
					case 8192 :
						Bcst_loc =  2; Bcst_est =  2; Comb_cst = 32; comb_loops =8; est_loops =16; loc_loops =7; threshold_loops =6; tolerance_loc =1e-8; tolerance_est =1e-8;
						break;
					case 16384 :
						Bcst_loc =  4; Bcst_est =  4; Comb_cst = 32; comb_loops =8; est_loops =10; loc_loops =6; threshold_loops =5; tolerance_loc =1e-8; tolerance_est =1e-8;
						break;
					case 32768 :
						Bcst_loc =  4; Bcst_est =  2; Comb_cst = 64; comb_loops =4; est_loops = 8; loc_loops =5; threshold_loops =4; tolerance_loc =1e-8; tolerance_est =1e-8;
						break;
					case 65536 :
						Bcst_loc =  4; Bcst_est =  2; Comb_cst =128; comb_loops =6; est_loops =10; loc_loops =4; threshold_loops =2; tolerance_loc =1e-8; tolerance_est =1e-8;
						break;
					case 131072 :
						Bcst_loc =  1; Bcst_est =  1; Comb_cst =  8; comb_loops =2; est_loops =12; loc_loops =4; threshold_loops =3; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 262144 :
						Bcst_loc =  1; Bcst_est =  1; Comb_cst =  8; comb_loops =2; est_loops =14; loc_loops =5; threshold_loops =4; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 524288 :
						Bcst_loc =0.5; Bcst_est =0.5; Comb_cst =  8; comb_loops =1; est_loops =10; loc_loops =4; threshold_loops =3; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 1048576 :
						Bcst_loc =0.5; Bcst_est =0.5; Comb_cst =  8; comb_loops =2; est_loops =12; loc_loops =4; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 2097152 :
						Bcst_loc =0.5; Bcst_est =0.2; Comb_cst =  8; comb_loops =1; est_loops =10; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 4194304 :
						Bcst_loc =0.5; Bcst_est =0.2; Comb_cst =  8; comb_loops =1; est_loops = 8; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 8388608 :
						Bcst_loc =0.5; Bcst_est =0.2; Comb_cst =  8; comb_loops =1; est_loops = 8; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 16777216 :
						Bcst_loc =0.5; Bcst_est =0.2; Comb_cst = 16; comb_loops =1; est_loops = 8; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					}
		}
		else{
				 switch(N){
					case 8192 :
						Bcst_loc =2; Bcst_est =  2; Comb_cst =1; comb_loops =1; est_loops =16; loc_loops =7; threshold_loops =6; tolerance_loc =1e-8; tolerance_est =1e-8;
						break;
					case 16384 :
						Bcst_loc =4; Bcst_est =  4; Comb_cst =1; comb_loops =1; est_loops =10; loc_loops =6; threshold_loops =5; tolerance_loc =1e-8; tolerance_est =1e-8;
						break;
					case 32768 :
						Bcst_loc =4; Bcst_est =  2; Comb_cst =1; comb_loops =1; est_loops = 8; loc_loops =5; threshold_loops =4; tolerance_loc =1e-8; tolerance_est =1e-8;
						break;
					case 65536 :
						Bcst_loc =4; Bcst_est =  2; Comb_cst =1; comb_loops =1; est_loops = 8; loc_loops =5; threshold_loops =4; tolerance_loc =1e-8; tolerance_est =1e-8;
						break;
					case 131072 :
						Bcst_loc =2; Bcst_est =  1; Comb_cst =1; comb_loops =1; est_loops =10; loc_loops =5; threshold_loops =4; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 262144 :
						Bcst_loc =2; Bcst_est =0.5; Comb_cst =1; comb_loops =1; est_loops =14; loc_loops =4; threshold_loops =3; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 524288 :
						Bcst_loc =1; Bcst_est =0.5; Comb_cst =1; comb_loops =1; est_loops =12; loc_loops =5; threshold_loops =4; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 1048576 :
						Bcst_loc =2; Bcst_est =0.5; Comb_cst =1; comb_loops =1; est_loops =12; loc_loops =4; threshold_loops =3; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 2097152 :
						Bcst_loc =2; Bcst_est =0.2; Comb_cst =1; comb_loops =1; est_loops =15; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 4194304 :
						Bcst_loc =4; Bcst_est =0.2; Comb_cst =1; comb_loops =1; est_loops =10; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 8388608 :
						Bcst_loc =2; Bcst_est =0.2; Comb_cst =1; comb_loops =1; est_loops = 8; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;
					case 16777216 :
						Bcst_loc =4; Bcst_est =0.2; Comb_cst =1; comb_loops =1; est_loops = 8; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1e-8;
						break;

					}
     	  }

		return;
  }


  void get_expermient_vs_K_parameters(int K, bool WITH_COMB, double &Bcst_loc, double &Bcst_est,  double &Comb_cst,  int &loc_loops, 
                                      int &est_loops,  int &threshold_loops, int &comb_loops, double &tolerance_loc, double &tolerance_est){

        if(WITH_COMB){

				 switch(K){
					case 50 :
						Bcst_loc =0.5; Bcst_est =0.2; Comb_cst = 16; comb_loops =1; est_loops =10; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1.0e-8;
						break;
					case 100 :
						Bcst_loc =0.5; Bcst_est =0.2; Comb_cst = 16; comb_loops =1; est_loops =12; loc_loops =4; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1.0e-8;
						break;
					case 200 :
						Bcst_loc =  0.5; Bcst_est =  0.5; Comb_cst = 32; comb_loops =1; est_loops = 8; loc_loops =4; threshold_loops =3; tolerance_loc =1e-6; tolerance_est =0.5e-8;
						break;
					case 500 :
						Bcst_loc =0.5; Bcst_est =0.5; Comb_cst = 64; comb_loops =1; est_loops =10; loc_loops =4; threshold_loops =3; tolerance_loc =1e-6; tolerance_est =0.5e-8;
						break;
					case 1000 :
						Bcst_loc =  1; Bcst_est =  1; Comb_cst =128; comb_loops =3; est_loops =12; loc_loops =4; threshold_loops =3; tolerance_loc =1e-6; tolerance_est =0.5e-8;
						break;
					case 2000 :
						Bcst_loc =  1; Bcst_est =  1; Comb_cst =512; comb_loops =3; est_loops =16; loc_loops =4; threshold_loops =3; tolerance_loc =1e-7; tolerance_est =0.2e-8;
						break;
					case 2500 :
						Bcst_loc =  1; Bcst_est =  1; Comb_cst =512; comb_loops =3; est_loops =16; loc_loops =4; threshold_loops =3; tolerance_loc =1e-7; tolerance_est =0.2e-8;
						break;
					case 4000 :
						Bcst_loc =  1; Bcst_est =  2; Comb_cst =512; comb_loops =3; est_loops =14; loc_loops =8; threshold_loops =7; tolerance_loc =1e-8; tolerance_est =0.5e-8;
						break;
					}
		}
		else{
				 switch(K){
					case 50 :
						Bcst_loc =4; Bcst_est =0.2; Comb_cst =1; comb_loops =1; est_loops =10; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1.0e-8;
						break;
					case 100 :
						Bcst_loc =2; Bcst_est =0.2; Comb_cst =1; comb_loops =1; est_loops =12; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =1.0e-8;
						break;
					case 200 :
						Bcst_loc =4; Bcst_est =0.5; Comb_cst =1; comb_loops =1; est_loops =10; loc_loops =3; threshold_loops =2; tolerance_loc =1e-6; tolerance_est =0.5e-8;
						break;
					case 500 :
						Bcst_loc =2; Bcst_est =  1; Comb_cst =1; comb_loops =1; est_loops =12; loc_loops =4; threshold_loops =3; tolerance_loc =1e-6; tolerance_est =0.5e-8;
						break;
					case 1000 :
						Bcst_loc =2; Bcst_est =  1; Comb_cst =1; comb_loops =1; est_loops =12; loc_loops =5; threshold_loops =4; tolerance_loc =1e-6; tolerance_est =1.0e-8;
						break;
					case 2000 :
						Bcst_loc =2; Bcst_est =  1; Comb_cst =1; comb_loops =1; est_loops =16; loc_loops =5; threshold_loops =4; tolerance_loc =1e-7; tolerance_est =0.5e-8;
						break;
					case 2500 :
						Bcst_loc =2; Bcst_est =  1; Comb_cst =1; comb_loops =1; est_loops =16; loc_loops =5; threshold_loops =4; tolerance_loc =1e-7; tolerance_est =0.5e-8;
						break;
					case 4000 :
						Bcst_loc =2; Bcst_est =  2; Comb_cst =1; comb_loops =1; est_loops =14; loc_loops =6; threshold_loops =5; tolerance_loc =1e-8; tolerance_est =1.0e-8;
						break;
			    }
     	  }

		return;
}

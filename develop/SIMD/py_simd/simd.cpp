
#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif


using namespace std;

float add_flt(float a, float b){
    return a + b;
}

void cpp_add_at32(float* S_klm, int* k, int* l, int* m, float* values, int Nv, int NwG, int NwL, int Nlines){
    for (int i=0; i < Nlines; i++){
        S_klm[NwG * NwL * k[i] + NwL * l[i] + m[i]] += values[i];  
    }
}
    

void cpp_calc_matrix_avx(float* database, float* S_klm, 
                        float v_min, float log_wG_min, float log_wL_min,
                        float dv,    float dxG,        float dxL,
                        int Nv,      int NwG,          int NwL, 
                        int Nlines) {

    __m256 vf_mins = { 0.0,v_min,log_wG_min,log_wL_min,0.0,v_min,log_wG_min,log_wL_min };
    __m256 vf_nds = { 1.0,dv,dxG,dxL,1.0,dv,dxG,dxL };
    __m256 vf_ones = { 1, 1, 1, 1, 1, 1, 1, 1 };
    __m256i vi_aptr = _mm256_set_epi32(1, NwL, NwL * NwG, 0, 1, NwL, NwL * NwG, 0);
    __m256i vi_wptr = _mm256_set_epi32((NwG + 1) * NwL, NwG * NwL, NwL, 0, (NwG + 1) * NwL, NwG * NwL, NwL, 0);
    __m256i vi_rptr = _mm256_set_epi32((NwG + 1) * NwL + 1, (NwG + 1) * NwL, NwG * NwL + 1, NwG * NwL, NwL + 1, NwL, 1, 0);

    __m256  vf_temp0;
    __m256  vf_SwGL;
    __m256  vf_klm;
    __m256i vi_klm0;
    __m256i vi_addr;
    __m256  vf_a0;
    __m256  vf_a1;
    __m256  vf_S;

    __m256i  index;

    for (int i = 0; i < 4*Nlines; i += 8) {
 
        // Store float k,l,m values in v_temp0:
        vf_klm = _mm256_load_ps(&database[i]);   //load data
        vf_klm = _mm256_sub_ps(vf_klm, vf_mins); //subtract min's
        vf_klm = _mm256_div_ps(vf_klm, vf_nds);  //divide by d's //Can be single FMA
        //vf_klm = _mm256_fmsub_ps(vf_klm, vf_inv_nds, vf_div_mins);

        // calculate the address of the 2x2 bins we're going to modfiy data.
        // the last x2 comes from the fact that on the lowest hierarchy, two bins
        // are always adjacent.
        vi_klm0 = _mm256_cvttps_epi32(vf_klm); // calc base indices
        vi_addr = _mm256_mullo_epi32(vi_klm0, vi_aptr); //multiply by array size
        vi_addr = _mm256_hadd_epi32(vi_addr, vi_addr); //sum elements
        vi_addr = _mm256_hadd_epi32(vi_addr, vi_addr); //base index for line 1 at 0-3; base index for line 2 at 4-7.
        vi_addr = _mm256_add_epi32(vi_addr, vi_wptr); //add relative indices
        _mm256_store_si256(&index, vi_addr); //store indices in normal register needed for write addresses
 
        // calc weights av,awG,awL:
        vf_a0 = _mm256_cvtepi32_ps(vi_klm0);   //convert indices to float
        vf_a1 = _mm256_sub_ps(vf_klm, vf_a0);  //calc a1 = i - i0
        vf_a0 = _mm256_sub_ps(vf_ones, vf_a1); //calc a0 = 1 - a1

        vf_SwGL  = _mm256_permute_ps(vf_klm, 0x00);   // |S|S|S|S||S|S|S|S|                        (lines |1111|2222|)
        vf_temp0 = _mm256_unpackhi_ps(vf_a0, vf_a1); // |awG1|awG0|awL1|awL0|awG1|awG0|awL1|awL0| (lines |1111|2222|)
        
        vf_klm  = _mm256_permute_ps(vf_temp0, 0xEE);// |awL1|awL0|awL1|awL0|awL1|awL0|awL1|awL0| (lines |1111|2222|) 
        vf_SwGL = _mm256_mul_ps(vf_klm, vf_SwGL); // |S*awL1|S*awL0|S*awL1|S*awL0||S*awL1|S*awL0|S*awL1|S*awL0|   

        vf_klm  = _mm256_permute_ps(vf_temp0, 0x50); // |awG1|awG1|awG0|awG0|awG1|awG1|awG0|awG0| (lines |1111|2222|) 
        vf_SwGL = _mm256_mul_ps(vf_klm, vf_SwGL); // |S*awG1*awL1|S*awG1*awL0|S*awG0*awL1|S*awG0*awL0||S*awG1*awL1|S*awG1*awL0|S*awG0*awL1|S*awG0*awL0|  


        // continue with line 1 only:
        vf_S = _mm256_i32gather_ps(&S_klm[index.m256i_i32[0]], vi_rptr, 4); // Gather the 8 values of S
        vf_temp0 = _mm256_permute2f128_ps(vf_a0, vf_a1, 0x20);              // |av0|av0|av0|av0|av1|av1|av1|av1| (line 1)
        vf_temp0 = _mm256_permute_ps(vf_temp0, 0x55);                       // av            for line 1
        vf_klm = _mm256_permute2f128_ps(vf_SwGL, vf_SwGL, 0x20);            // S0*awG*awL    for line 1 //could go earlier //perhaps better operation?
        vf_S = _mm256_fmadd_ps(vf_klm, vf_temp0, vf_S);                     // Add the new values S = S + S0*awG*awL*av
  
        _mm_storel_pi(reinterpret_cast<__m64*>(&S_klm[index.m256i_i32[0]]), _mm256_castps256_ps128(vf_S)); //Store S000,S001
        _mm_storeh_pi(reinterpret_cast<__m64*>(&S_klm[index.m256i_i32[1]]), _mm256_castps256_ps128(vf_S)); //Store S010,S011
        vf_S = _mm256_permute2f128_ps(vf_S, vf_S, 0x01); //Is there really no single operation for this?
        _mm_storel_pi(reinterpret_cast<__m64*>(&S_klm[index.m256i_i32[2]]), _mm256_castps256_ps128(vf_S)); //Store S100,S101
        _mm_storeh_pi(reinterpret_cast<__m64*>(&S_klm[index.m256i_i32[3]]), _mm256_castps256_ps128(vf_S)); //Store S110,S111


        // on to line 2:
        vf_S = _mm256_i32gather_ps(&S_klm[index.m256i_i32[4]], vi_rptr, 4); // Gather the 8 values of S 
        vf_temp0 = _mm256_permute2f128_ps(vf_a0, vf_a1, 0x31);              // |av0|av0|av0|av0|av1|av1|av1|av1| (line 2)
        vf_temp0 = _mm256_permute_ps(vf_temp0, 0x55);                       // av            for line 2
        vf_klm = _mm256_permute2f128_ps(vf_SwGL, vf_SwGL, 0x31);            // S0*awG*awL    for line 2 //could go earlier
        vf_S = _mm256_fmadd_ps(vf_klm, vf_temp0, vf_S);                     // Add the new values S = S + S0*awG*awL*av

        _mm_storel_pi(reinterpret_cast<__m64*>(&S_klm[index.m256i_i32[4]]), _mm256_castps256_ps128(vf_S)); //Store S000,S001
        _mm_storeh_pi(reinterpret_cast<__m64*>(&S_klm[index.m256i_i32[5]]), _mm256_castps256_ps128(vf_S)); //Store S010,S011
        vf_S = _mm256_permute2f128_ps(vf_S, vf_S, 0x01); //Is there really no single operation for this?
        _mm_storel_pi(reinterpret_cast<__m64*>(&S_klm[index.m256i_i32[6]]), _mm256_castps256_ps128(vf_S)); //Store S100,S101
        _mm_storeh_pi(reinterpret_cast<__m64*>(&S_klm[index.m256i_i32[7]]), _mm256_castps256_ps128(vf_S)); //Store S110,S111
        
    }
}
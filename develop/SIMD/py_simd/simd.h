
using namespace std;

float add_flt(float a, float b);

void cpp_add_at32(float* S_klm, int* k, int* l, int* m, float* values, int Nv, int NwG, int NwL, int Nlines);

void cpp_calc_matrix(float* database, float* S_klm, 
                        float v_min, float log_wG_min, float log_wL_min,
                        float dv,    float dxG,        float dxL,
                        int Nv,      int NwG,          int NwL, 
                        int Nlines);
                        
void cpp_calc_matrix_avx(float* database, float* S_klm, 
        float v_min, float log_wG_min, float log_wL_min,
        float dv,    float dxG,        float dxL,
        int Nv,      int NwG,          int NwL, 
        int Nlines);
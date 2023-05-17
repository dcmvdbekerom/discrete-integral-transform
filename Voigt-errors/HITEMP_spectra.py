import numpy as np
from time import perf_counter
from scipy.constants import h,c,k,N_A,pi
c_cm = c*100
c2 = h * c_cm / k

w1  = 1354.31
w2  =  672.85
w3  = 2396.32
d1  =    1.00
d2  =    2.00
d3  =    1.00
x11 =   -2.93
x12 =   -4.61
x13 =  -19.82
x22 =    1.35
x23 =  -12.31
x33 =  -12.47
l2l2=   -0.97
B   =    3.9020e-01

def aligned_array(shape,alignment,dtyp,zeros = False,show_proof = False):
    dtype = np.dtype(dtyp)
    nbytes = np.prod(shape) * dtype.itemsize
    f_init = (np.zeros if zeros else np.empty)
    buf = f_init(nbytes + alignment, dtype=np.uint8)
    start_index = -buf.ctypes.data % alignment
    arr = buf[start_index:start_index + nbytes].view(dtype).reshape(shape)
    if show_proof:
        print(arr.ctypes.data % alignment)
    return arr

def import_HITRAN(fnames):
    iso,v0,A21,gs,El,na,da,gu = [],[],[],[],[],[],[],[]
    
    fnames = ([fnames] if type(fnames) == str else sorted(fnames))
    
    for fname in fnames:
        print("Loading " + fname + "...")
        with open(fname,'r') as f:
            for line in f:
                iso.append(  int(line[  2:  3])) #iso
                v0 .append(float(line[  3: 15])) #v0
                A21.append(float(line[ 25: 35])) #A21
                gs .append(float(line[ 40: 45])) #gamma_self
                El .append(float(line[ 45: 55])) #Elow
                na .append(float(line[ 55: 59])) #n_air
                da .append(float(line[ 59: 67])) #d_air
                gu .append(float(line[146:153])) #g_up
            
    return np.array([iso,v0,A21,gs,El,na,da,gu])

def load_HITRAN(fname):
    data = import_HITRAN(fname)
    data = data[:,data[0]<=3] # select only first three isotopes
    iso,v0,A21,gs,El,na,da,gu = data

    iso_int = iso.astype(int)-1
    
    Mm   = (np.array([44,45,46])* 1e-3 / N_A)[iso_int]
    f_ab =  np.array([ 0.98420, 0.01106, 0.0039471])[iso_int]
    gr   =  np.array([0.5,1.0,1.0])[iso_int]
    Eu   = El + v0
    S0 = f_ab * gu * A21 / (8*pi*c_cm*v0**2)
    
    log_2gs  = np.log(2*gs)                                      #vector
    log_2vMm = np.log(2*v0) + 0.5*np.log(2*k*np.log(2)/(c**2*Mm))#vector
                  
    return np.array([v0,da,S0,El,Eu,log_2gs,na,log_2vMm,gr],dtype = np.float32)


def init_database(file_list):
    global data
    try:
        data = np.load('CO2_hitemp.npy')
        
    except(FileNotFoundError):
        data = load_HITRAN(file_list)
        print("Saving database as numpy file...")
        np.save('CO2_hitemp.npy',data)

def calc_Ev12(v1,v2,l2):
    E0  =  0.5*w1*d1 + 0.5*w2*d2 + x11*(0.5*d1)**2 + 0.25*x12*d1*d2 + 0.25*x22*d2**2
    return w1*(v1+0.5*d1) + w2*(v2+0.5*d2) + x11*(v1+0.5*d1)**2 + x12*(v1+0.5*d1)*(v2+0.5*d2) + x22*(v2+0.5*d2)**2 + l2l2*l2**2 - E0

def calc_Ev3(v3):
    E0 = 0.5*w3*d3 + 0.25*x33*d3**2
    return w3*(v3+0.5*d3) + x33*(v3+0.5*d3)**2 - E0
    
def calc_Evc(v1,v2,v3):
    E0 = 0.25*x13*d1*d3 + 0.25*x23*d2*d3
    return x13*(v1+0.5*d1)*(v3+0.5*d3) + x23*(v2+0.5*d2)*(v3+0.5*d3) - E0

def Qv12(T12):
    return (1-np.exp(-c2*calc_Ev12(1,0,0)/T12))**-1 * (1-np.exp(-c2*calc_Ev12(0,1,0)/T12))**-2

def Qv3(T3):
    return (1-np.exp(-c2*calc_Ev3(1)/T3))**-1 

def Qv(T12,T3):
    return Qv12(T12)*Qv3(T3)

def Qr(Tr): #McDowell 1978
    return Tr/(c2*B)*np.exp(c2*B/(3*Tr))

def calc_stick_spectrum(p,T):
    global data
    v0,da,S0,El,Eu,log_2gs,na,log_2vMm,gr = data

    #Each iteration, but only on scalar:
    c2T       = -h*c_cm/(k*T)  #scalar
    log_p     = np.log(p)      #scalar
    log_rT    = np.log(296./T) #scalar
    hlog_T    = 0.5*np.log(T)  #scalar
    N         = p*1e5 / (1e6 * k * T) #scalar

    
    v_dat      = v0 + p*da
    log_wG_dat = log_2vMm + hlog_T #minmax can be determined at init
    log_wL_dat = log_2gs + log_p + na*log_rT #minmax function can be determined at init
    I_dat      = N * S0 * (np.exp(c2T*El) - np.exp(c2T*Eu)) / np.log(10) / (gr*Qr(T)*Qv(T,T))
    #I_dat      = N * S0 * (np.exp(-h*c_cm*El/(k*T)) - np.exp(-h*c_cm*Eu/(k*T))) / np.log(10) / (0.5*Qr(T)*Qv(T,T))

    return (v_dat,log_wG_dat,log_wL_dat,I_dat)

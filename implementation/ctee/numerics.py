"""Small, inspectable numerical algorithms: QR, constrained fits, and radix-2 FFT."""
from functools import lru_cache
from itertools import combinations
import numpy as np

def norm(x):
    return float(np.sqrt(np.sum(np.abs(x)**2)))

def qr_factor(a, tolerance=1e-12):
    """Twice-reorthogonalised modified Gram-Schmidt for a real tall matrix."""
    a=np.asarray(a,dtype=float)
    if a.ndim!=2 or a.shape[0]<a.shape[1] or not np.isfinite(a).all():raise ValueError('Invalid QR matrix')
    m,n=a.shape;q=np.zeros((m,n));r=np.zeros((n,n));limit=tolerance*max(norm(a),1.)
    for j in range(n):
        v=a[:,j].copy()
        for _ in range(2):
            for i in range(j):
                projection=float(q[:,i]@v);r[i,j]+=projection;v-=projection*q[:,i]
        length=norm(v)
        if length<=limit:raise ValueError('Rank-deficient design matrix')
        q[:,j]=v/length;r[j,j]=length
    return q,r

def solve_upper(r,b):
    r=np.asarray(r,dtype=float);b=np.asarray(b,dtype=float)
    vector=b.ndim==1
    if vector:b=b[:,None]
    if r.shape!=(len(b),len(b)):raise ValueError('Triangular solve dimensions differ')
    x=np.zeros_like(b)
    for i in reversed(range(len(r))):
        if abs(r[i,i])<1e-15:raise ValueError('Singular triangular system')
        x[i]=(b[i]-r[i,i+1:]@x[i+1:])/r[i,i]
    return x[:,0] if vector else x

def least_squares(a,b):
    q,r=qr_factor(a)
    return solve_upper(r,q.T@np.asarray(b,dtype=float))

def projection_matrix(a):
    q,r=qr_factor(a)
    return solve_upper(r,q.T)

def nonnegative_fit(a,b,tolerance=1e-10):
    """Exhaust all active subsets: only three R/L/C coefficients are needed."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.ndim!=2 or b.shape!=(len(a),) or a.shape[1]>6 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid small nonnegative fit')
    n=a.shape[1];best=np.zeros(n);cost=float(b@b)
    for size in range(1,n+1):
        for active in combinations(range(n),size):
            try:solution=least_squares(a[:,active],b)
            except ValueError:continue
            if (solution < -tolerance).any():continue
            candidate=np.zeros(n);candidate[list(active)]=np.maximum(solution,0.)
            residual=a@candidate-b;value=float(residual@residual)
            if value<cost:best,cost=candidate,value
    return best,float(np.sqrt(cost))

def regression_slope(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);dx=x-x.mean();dy=y-y.mean()
    denominator=float(dx@dx)
    if x.shape!=y.shape or len(x)<2 or denominator<=1e-15:raise ValueError('Insufficient regression excitation')
    return float(dx@dy/denominator)

@lru_cache(maxsize=16)
def _fft_plan(n):
    indices=np.arange(n);reverse=np.zeros(n,dtype=int);working=indices.copy()
    for _ in range(n.bit_length()-1):reverse=(reverse<<1)|(working&1);working>>=1
    twiddles={width:np.exp(-2j*np.pi*np.arange(width//2)/width) for width in [1<<i for i in range(1,n.bit_length())]}
    return reverse,twiddles

def fft_radix2(samples):
    """Explicit iterative Cooley-Tukey butterflies; zero-pad to next power of two."""
    x=np.asarray(samples,dtype=complex)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('FFT needs a finite nonempty vector')
    n=1<<(len(x)-1).bit_length();padded=np.zeros(n,dtype=complex);padded[:len(x)]=x
    reverse,twiddles=_fft_plan(n);out=padded[reverse].copy()
    for width,twiddle in twiddles.items():
        block=out.reshape(-1,width);half=width//2;even=block[:,:half].copy();odd=block[:,half:]*twiddle
        block[:,:half]=even+odd;block[:,half:]=even-odd
    return out

def fft_signature(samples,sample_rate,fundamental):
    x=np.asarray(samples);window=.5-.5*np.cos(2*np.pi*np.arange(len(x))/max(len(x)-1,1))
    spectrum=fft_radix2(x*window);power=np.abs(spectrum[:len(spectrum)//2+1])**2
    probabilities=power/max(float(power.sum()),1e-30);frequencies=np.arange(len(power))*sample_rate/len(spectrum)
    high=float(probabilities[frequencies>8*fundamental].sum())
    positive=probabilities>0;entropy=float(-np.sum(probabilities[positive]*np.log(probabilities[positive]))/np.log(max(len(power),2)))
    return high,entropy

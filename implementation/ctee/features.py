"""Electrical evidence computed only from the selected measured waveforms."""
import numpy as np
from functools import lru_cache
from .numerics import nonnegative_fit, projection_matrix, regression_slope, fft_signature, norm
from .simulator import HARMONICS,SAMPLE_RATE,COMPOSITIONS

PER_PROBE_NAMES=("log_vrms","log_irms","active_power","true_pf","sin_phase","cos_phase",
                 "log_abs_z1","z1_real_fraction","z1_imag_fraction","thd_v","thd_i",
                 "crest_i","signed_vi_area","available")
BASE_NAMES=[f"probe{p}_{s}" for p in range(5) for s in PER_PROBE_NAMES]
EXTRA_NAMES=["current_voltage_slope","power_voltage_slope","voltage_span","slope_valid",
             "R_fraction","L_fraction","C_fraction","spectral_fit_error","spectral_valid",
             *[f"physics_{x}" for x in COMPOSITIONS],"phase_leading_fraction", "fft_high_frequency_ratio", "fft_entropy",
             "startup_peak_ratio","startup_early_rms_ratio","startup_late_rms_ratio",
             "startup_ringing_ratio","startup_power_fraction","startup_available"]
FEATURE_NAMES=BASE_NAMES+EXTRA_NAMES
N_BASE=len(BASE_NAMES)
N_FEATURES=len(FEATURE_NAMES)

@lru_cache(maxsize=64)
def _basis(n,frequency,sample_rate):
    t=np.arange(n)/sample_rate
    # Joint sine/cosine least squares works with arbitrary phase and noncoherent f.
    arg=2*np.pi*np.outer(t,frequency*HARMONICS)
    design=np.column_stack([np.cos(arg),np.sin(arg),np.ones(len(t))])
    return design,projection_matrix(design)

def phasors(wave,frequency,sample_rate=SAMPLE_RATE):
    design,pinv=_basis(wave.shape[-1],float(frequency),sample_rate)
    coef=pinv@wave.T
    p=(coef[:4]-1j*coef[4:8])/np.sqrt(2)
    residual=wave.T-design@coef
    se=np.sqrt(np.mean(residual**2,axis=0)/wave.shape[-1])
    return p.T,se

def spectral_evidence(vp,ip,se):
    valid=(np.abs(vp)>max(.008*abs(vp[0]),5*se[0])) & (np.abs(ip)>5*se[1])
    if valid.sum()<3:
        return np.zeros(7),np.zeros(3),1.,False
    h=HARMONICS[valid].astype(float)
    z=vp[valid]/ip[valid]
    # Linear series-RLC fit in Ohms: Re Z=R; Im Z=h*XL-XC/h.
    a=np.zeros((2*len(h),3));a[:len(h),0]=1
    a[len(h):,1]=h;a[len(h):,2]=-1/h
    b=np.r_[z.real,z.imag]
    weights=np.minimum(np.abs(ip[valid])**2/np.maximum(np.abs(vp[valid]),1e-10),1e6)
    weights=weights/max(weights.max(),1e-12)
    coef,_=nonnegative_fit(a*np.r_[weights,weights][:,None],b*np.r_[weights,weights])
    fractions=coef/max(coef.sum(),1e-12)
    residual=float(norm((a@coef-b)*np.r_[weights,weights])/max(norm(b*np.r_[weights,weights]),1e-12))
    # Smooth evidence for component presence, not hidden simulator parameters.
    presence=np.clip((fractions-.015)/.035,.001,.999)
    scores=[]
    for label in COMPOSITIONS:
        mask=np.array([x in label for x in 'RLC'])
        scores.append(np.sum(np.log(np.where(mask,presence,1-presence))))
    scores=np.asarray(scores);scores-=scores.max()
    return scores,fractions,residual,True

def extract(waves,frequencies,selected=None,sample_rate=SAMPLE_RATE,startup=None):
    if selected is None:selected=range(len(waves))
    selected=sorted(set(int(p) for p in selected))
    if not selected or any(p not in range(5) for p in selected):raise ValueError("Choose at least one valid probe")
    if np.asarray(waves).shape[:2]!=(5,2):raise ValueError("Expected [5 probes, 2 channels, samples]")
    features=np.zeros(N_FEATURES,dtype=np.float32)
    measurements={};best=None
    for p in selected:
        wave=np.asarray(waves[p],dtype=float)
        if not np.isfinite(wave).all():raise ValueError("Measurements contain non-finite values")
        if wave.shape[-1]<100:raise ValueError("Too few samples")
        v,i=wave
        vr,ir=np.sqrt(np.mean(wave**2,axis=1))
        if min(vr,ir)<1e-9:raise ValueError("Voltage/current is too small for identification")
        power=float(np.mean(v*i));pf=np.clip(power/(vr*ir),-1,1)
        pv,pi=phasors(wave,float(frequencies[p]),sample_rate)
        # phasors() returns a channel matrix and standard errors.
        vp,ip=pv
        z=vp[0]/ip[0] if abs(ip[0])>1e-10 else 0j
        phi=np.angle(z);zabs=max(abs(z),1e-9)
        thdv=float(norm(vp[1:])/max(abs(vp[0]),1e-12))
        thdi=float(norm(ip[1:])/max(abs(ip[0]),1e-12))
        area=float(np.sum(v*np.roll(i,-1)-i*np.roll(v,-1))/(len(v)*vr*ir))
        block=[np.log(vr),np.log(ir),power,pf,np.sin(phi),np.cos(phi),np.log(zabs),z.real/zabs,z.imag/zabs,
               thdv,thdi,np.max(abs(i))/ir,area,1.]
        features[p*14:(p+1)*14]=block
        measurements[p]=(vr,ir,power,pf,phi)
        evidence=spectral_evidence(vp,ip,pi)
        # Select by excited spectral coverage then fit error, independent of labels.
        if evidence[3] and (best is None or evidence[2]<best[2]):best=evidence
    slope_i=slope_p=span=0.;slope_valid=False
    amplitude=[measurements[p] for p in selected if p<3]
    if len(amplitude)>=2:
        arr=np.array(amplitude);lv=np.log(arr[:,0]);span=float(np.ptp(lv))
        if span>.025:
            slope_i=regression_slope(lv,np.log(arr[:,1]))
            if np.min(arr[:,2])>1e-8:slope_p=regression_slope(lv,np.log(arr[:,2]))
            slope_valid=True
    scores=np.zeros(7);fractions=np.zeros(3);error=1.;spec=False
    if best is not None:scores,fractions,error,spec=best
    extras=[np.clip(slope_i,-4,4),np.clip(slope_p,-4,4),span,float(slope_valid),*fractions,error,float(spec),*scores,
            np.mean([m[4]<0 for m in measurements.values()])]
    extras.extend(fft_signature(waves[selected[0],1],sample_rate,float(frequencies[selected[0]])))
    dynamics=np.zeros(6)
    if startup is not None:
        sv,si=np.asarray(startup,dtype=float)
        n=len(si);late=max(np.sqrt(np.mean(si[-n//4:]**2)),1e-12)
        dynamics=np.array([np.max(abs(si))/late,np.sqrt(np.mean(si[:n//8]**2))/late,
                           np.sqrt(np.mean(si[n//4:n//2]**2))/late,
                           np.sqrt(np.mean(np.diff(si)**2))/late,
                           np.mean(sv*si)/max(np.sqrt(np.mean(sv**2))*late,1e-12),1.])
    extras.extend(dynamics)
    features[N_BASE:]=extras
    return features

def physics_logits(features):
    x=np.atleast_2d(features)
    slope=x[:,N_BASE]
    valid=x[:,N_BASE+3]>0.5
    b=-((slope[:,None]-np.array([0.,-1.,1.])[None,:])/.3)**2
    b=np.clip(b,-20,0)*valid[:,None]
    c=x[:,N_BASE+9:N_BASE+16].copy()
    return b,c

def evidence_summary(features):
    x=np.asarray(features)
    return {"current_voltage_slope":float(x[N_BASE]),"power_voltage_slope":float(x[N_BASE+1]),
            "voltage_span":float(x[N_BASE+2]),"slope_valid":bool(x[N_BASE+3]),
            "relative_R":float(x[N_BASE+4]),"relative_L":float(x[N_BASE+5]),"relative_C":float(x[N_BASE+6]),
            "spectral_fit_error":float(x[N_BASE+7]),"spectral_valid":bool(x[N_BASE+8])}


"""Explicit steady-state controlled-source emulator; never hidden component labels.

Terminal source V -- controlled series voltage Uc -- passive series RLC branch.
Vcircuit = V + Uc = gain * V. In CZ, gain=1 and Uc=0. In CC/CP a
four-quadrant ideal voltage controller regulates RMS current/active power.
This is a behavioural AC emulator, not a switched-converter hardware model.
"""
from dataclasses import dataclass, asdict
import numpy as np

BEHAVIOURS = ("CC", "CP", "CZ")
COMPOSITIONS = ("R", "L", "C", "RL", "RC", "LC", "RLC")
PROBES = ("Nominal", "Voltage -5%", "Voltage +5%", "Harmonic probe", "80 Hz probe")
SAMPLE_RATE = 5000
DURATION = 0.24
N_SAMPLES = int(SAMPLE_RATE * DURATION)
HARMONICS = np.array([1, 3, 5, 7])
SUPPORT = [(b,c) for c in COMPOSITIONS for b in BEHAVIOURS if b != "CP" or "R" in c]

@dataclass(frozen=True)
class Circuit:
    group_id: str
    composition: str
    resistance: float
    inductance: float
    capacitance: float
    parasitic_resistance: float
    phase: float
    source_harmonics: tuple

    def to_dict(self):
        return asdict(self)

def sample_circuit(composition, index, rng):
    if composition not in COMPOSITIONS:
        raise ValueError("Unknown circuit composition")
    # Comparable impedances across families prevent nominal-power shortcuts.
    r, xl, xc = np.exp(rng.uniform(np.log(6),np.log(65),3))
    if "L" in composition and "C" in composition:
        while abs(xl-xc) < 0.20*max(xl,xc):
            xc = np.exp(rng.uniform(np.log(6),np.log(65)))
    return Circuit(f"{composition}-{index:05d}",composition,
                   float(r if "R" in composition else 0),
                   float(xl/(2*np.pi*50) if "L" in composition else 0),
                   float(1/(2*np.pi*50*xc) if "C" in composition else 0),
                   float(rng.uniform(.005,.025)),float(rng.uniform(-np.pi,np.pi)),
                   tuple(rng.uniform(.001,.006,3)))

def impedance(circuit, frequency):
    f = np.asarray(frequency, dtype=float)
    if np.any(f<=0):
        raise ValueError("Frequency must be positive")
    z = np.full(f.shape,circuit.resistance+circuit.parasitic_resistance,dtype=complex)
    if circuit.inductance:
        z += 1j*2*np.pi*f*circuit.inductance
    if circuit.capacitance:
        z -= 1j/(2*np.pi*f*circuit.capacitance)
    return z

def supported(circuit, behaviour):
    return behaviour in BEHAVIOURS and (behaviour!="CP" or "R" in circuit.composition)

def clean_probe(circuit, behaviour, probe_id, frequency_shift=1.):
    if not supported(circuit,behaviour):
        raise ValueError("Unsupported pair: nonzero active-power CP requires a declared R branch")
    if probe_id not in range(len(PROBES)):
        raise ValueError("Unknown probe")
    t = np.arange(N_SAMPLES)/SAMPLE_RATE
    f = (80 if probe_id==4 else 50)*frequency_shift
    level = (1.,.95,1.05,1.,1.)[probe_id]
    hs = np.array([1.,*(circuit.source_harmonics if probe_id!=3 else (.10,.075,.05))])
    vp = level*hs*np.exp(1j*circuit.phase*HARMONICS)
    yp = 1/impedance(circuit,f*HARMONICS)
    raw_i = vp*yp
    basis = np.exp(2j*np.pi*f*np.outer(t,HARMONICS))
    v = np.sqrt(2)*np.real(basis@vp)
    passive_i = np.sqrt(2)*np.real(basis@raw_i)
    nominal_y = 1/impedance(circuit,np.array([50.]))[0]
    if behaviour=="CC":
        gain = abs(nominal_y)/np.sqrt(np.mean(passive_i**2))
    elif behaviour=="CP":
        p0 = nominal_y.real
        gain = p0/np.mean(v*passive_i)
    else:
        gain = 1.
    current = passive_i*gain
    # Physical internal/controller voltages and power are available for audit only.
    wave = np.stack([v,current],axis=0).astype(np.float32)
    return wave,{"sample_rate":SAMPLE_RATE,"frequency":f,"gain":float(gain),
                 "controller_active":behaviour!="CZ","probe_id":probe_id}

def corrupt(wave, rng, snr_db=None, condition="nominal"):
    out = wave.astype(float).copy()
    if condition=="sensor_shift":
        out *= np.array([1.025,.975])[:,None]
        out[1] = np.roll(out[1],1)
    if snr_db is not None:
        noise=rng.standard_normal(out.shape)
        if condition=="coloured":
            for t in range(1,noise.shape[-1]):noise[:,t]+=.65*noise[:,t-1]
        noise /= np.maximum(np.sqrt(np.mean(noise**2,axis=1,keepdims=True)),1e-12)
        sigma = np.sqrt(np.mean(wave.astype(float)**2,axis=1,keepdims=True))*10**(-snr_db/20)
        out += noise*sigma
    return out.astype(np.float32)

def episode(circuit, behaviour, seed=0, snr_db=None, condition="nominal"):
    rng=np.random.default_rng(seed)
    shift=1.2 if condition=="frequency_shift" else 1.
    waves=[];frequencies=[]
    for p in range(len(PROBES)):
        w,info=clean_probe(circuit,behaviour,p,shift)
        waves.append(corrupt(w,rng,snr_db,condition));frequencies.append(info['frequency'])
    return np.stack(waves),np.asarray(frequencies)

def transient(circuit,duration=.3,sample_rate=10000):
    """Zero-state response to a smooth sine startup, independently of mode labels."""
    t=np.arange(int(duration*sample_rate))/sample_rate
    v=np.sin(2*np.pi*50*t)*(1-np.exp(-t/.01))
    r=circuit.resistance+circuit.parasitic_resistance
    l,c=circuit.inductance,circuit.capacitance
    if not l and not c:return t,v,v/r
    dt=1/sample_rate;i=np.zeros(len(t))
    if l and c:
        # Implicit trapezoidal integration of [current, capacitor voltage].
        # Invert the 2x2 system explicitly; no ready-made ODE solver.
        m00=1+dt*r/(2*l);m01=dt/(2*l);m10=-dt/(2*c)
        determinant=m00-m01*m10
        inverse=np.array([[1.,-m01],[-m10,m00]])/determinant
        transition=inverse@np.array([[1-dt*r/(2*l),-dt/(2*l)],[dt/(2*c),1.]])
        drive=inverse@np.array([dt/(2*l),0.])
        f00,f01,f10,f11=transition.ravel();g0,g1=drive;current=capacitor=0.
        for k in range(1,len(t)):
            source=v[k-1]+v[k]
            new_current=f00*current+f01*capacitor+g0*source
            capacitor=f10*current+f11*capacitor+g1*source
            current=new_current;i[k]=current
    else:
        # Exact first-order response to linearly interpolated source samples.
        tau=l/r if l else r*c;alpha=dt/tau;decay=np.exp(-alpha)
        ratio=-np.expm1(-alpha)/alpha;previous_weight=ratio-decay;next_weight=1-ratio
        state=0.
        for k in range(1,len(t)):
            forcing=previous_weight*v[k-1]+next_weight*v[k]
            state=decay*state+(forcing/r if l else forcing)
            i[k]=state if l else (v[k]-state)/r
    return t,v,i


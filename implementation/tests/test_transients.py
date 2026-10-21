import numpy as np
from ctee.simulator import Circuit,transient

def test_series_rlc_satisfies_kirchhoff_equation():
    c=Circuit('equation','RLC',10.,.1,100e-6,.01,0.,(.001,.001,.001))
    t,v,i=transient(c,duration=.3,sample_rate=20000);dt=t[1]-t[0]
    charge=np.zeros_like(i);charge[1:]=np.cumsum((i[1:]+i[:-1])*dt/2)
    derivative=(i[2:]-i[:-2])/(2*dt)
    reconstructed=(c.resistance+c.parasitic_resistance)*i[1:-1]+c.inductance*derivative+charge[1:-1]/c.capacitance
    relative=np.sqrt(np.mean((v[1:-1]-reconstructed)**2))/np.sqrt(np.mean(v[1:-1]**2))
    assert relative<.001

def test_transient_converges_with_time_step():
    c=Circuit('convergence','RLC',10.,.1,100e-6,.01,0.,(.001,.001,.001))
    _,_,coarse=transient(c,sample_rate=5000)
    _,_,fine=transient(c,sample_rate=10000)
    _,_,reference=transient(c,sample_rate=20000)
    error_coarse=np.sqrt(np.mean((coarse-reference[::4])**2))
    error_fine=np.sqrt(np.mean((fine-reference[::2])**2))
    assert error_fine<error_coarse/2

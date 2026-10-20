"""Strict measured-waveform schema. No deserialisation or user code execution."""
import io
import numpy as np
import pandas as pd
from .simulator import SAMPLE_RATE

COLUMNS=['probe','time','voltage','current','frequency']

def export_csv(waves,frequencies,selected=None,sample_rate=SAMPLE_RATE):
    selected=range(5) if selected is None else selected
    frames=[]
    for p in selected:
        frames.append(pd.DataFrame({'probe':p,'time':np.arange(waves.shape[-1])/sample_rate,
                      'voltage':waves[p,0],'current':waves[p,1],'frequency':frequencies[p]}))
    return pd.concat(frames,ignore_index=True).to_csv(index=False)

def parse_csv(content):
    if isinstance(content,bytes):content=content.decode('utf-8-sig')
    if len(content)>10_000_000:raise ValueError('CSV is larger than 10 MB')
    df=pd.read_csv(io.StringIO(content))
    if not set(COLUMNS)<=set(df.columns):raise ValueError('Required columns: '+', '.join(COLUMNS))
    df=df[COLUMNS].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(df.to_numpy()).all():raise ValueError('All measurements must be finite numbers')
    if not df['probe'].isin(range(5)).all():raise ValueError('Probe IDs must be integers from 0 to 4')
    selected=sorted(df['probe'].astype(int).unique().tolist())
    if 0 not in selected:raise ValueError('Include nominal probe 0 as a reference')
    arrays={};fs={};sample_rates=[];lengths=[]
    for p,g in df.groupby('probe',sort=True):
        t=g['time'].to_numpy();dt=np.diff(t)
        if len(t)<100 or (dt<=0).any():raise ValueError('Each probe needs >=100 samples with increasing time')
        step=float(np.median(dt))
        if not np.allclose(dt,step,rtol=1e-3,atol=1e-8):raise ValueError('Sampling must be uniform; gaps are not silently interpolated')
        freq=g['frequency'].to_numpy()
        if np.ptp(freq)>1e-5 or not (30<=freq[0]<=150):raise ValueError('Frequency must be constant per probe and 30-150 Hz')
        if t[-1]-t[0]<4/freq[0]:raise ValueError('Record at least four source cycles per probe')
        rate=1/step
        if rate<20*freq[0]:raise ValueError('Sample rate is too low for the harmonic analysis')
        arrays[int(p)]=g[['voltage','current']].to_numpy().T
        fs[int(p)]=float(freq[0]);sample_rates.append(rate);lengths.append(len(g))
    if len(set(lengths))!=1 or not np.allclose(sample_rates,sample_rates[0],rtol=1e-3):
        raise ValueError('All probes must have equal lengths and the same sample rate')
    waves=np.zeros((5,2,lengths[0]),dtype=np.float32);frequencies=np.full(5,50.)
    for p in selected:waves[p]=arrays[p];frequencies[p]=fs[p]
    return waves,frequencies,selected,float(sample_rates[0])


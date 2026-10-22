"""Local demonstration; run with .venv_numpy/bin/python -m streamlit run app.py."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from threadpoolctl import threadpool_limits
from ctee.simulator import *
from ctee.inference import Predictor,trace_policy
from ctee.csv_io import export_csv,parse_csv
from ctee.runtime import DEFAULT_RUN
from ctee.parameters import parameter_summary,export_parameters,preview_edit

ROOT=Path(__file__).resolve().parent
RUN_ROOT=DEFAULT_RUN
st.set_page_config(page_title='CTEE LoadLab',page_icon='⚡',layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1450px}h1{letter-spacing:-1px}
[data-testid="stMetric"]{border:1px solid #d9e6eb;border-radius:12px;padding:16px}
</style>''',unsafe_allow_html=True)
st.title('⚡ CTEE LoadLab')
st.caption('Learned weights and physics trust · One shared encoder · Two electrical answers')
st.info('Research prototype: an ideal controlled-source simulator with a known series RLC branch. Results describe this benchmark; they do not establish real-appliance or hardware accuracy.')

@st.cache_resource
def predictor(revision):return Predictor(RUN_ROOT)

if not (RUN_ROOT/'artifacts/selected_model.npz').exists():
    st.warning('Train the model first: `.venv_numpy/bin/python -m ctee train`')
    st.stop()

pred=predictor((RUN_ROOT/'artifacts/selected_model.npz').stat().st_mtime_ns)
demo,results,method,parameters=st.tabs(['Classify a load','Measured comparisons','How it works','Weights and learning'])

def show_prediction(result,waves,frequencies,truth=None,allow_export=True,sample_rate=SAMPLE_RATE):
    decision=result['decision'] if 'decision' in result else result
    columns=st.columns(4)
    columns[0].metric('Behaviour',BEHAVIOURS[decision['behaviour']])
    columns[1].metric('Branch composition',COMPOSITIONS[decision['composition']])
    columns[2].metric('Minimum head confidence',f"{decision['confidence']:.1%}")
    columns[3].metric('Status','Accepted' if decision['accepted'] else 'Inconclusive')
    if not decision['accepted']:st.warning('Tentative labels. '+'; '.join(decision['reasons']))
    else:st.success('The confidence and electrical evidence gates passed for this scoped benchmark.')
    ev=decision['evidence'];left,right=st.columns(2)
    with left:
        pfig=make_subplots(rows=1,cols=2,subplot_titles=['Behaviour probabilities','Composition probabilities'])
        pfig.add_trace(go.Bar(x=BEHAVIOURS,y=decision['behaviour_probabilities'],marker_color='#087f82'),row=1,col=1)
        pfig.add_trace(go.Bar(x=COMPOSITIONS,y=decision['composition_probabilities'],marker_color='#2454a0'),row=1,col=2)
        pfig.update_yaxes(range=[0,1]);pfig.update_layout(height=320,showlegend=False,margin=dict(l=20,r=20,t=35,b=20))
        st.plotly_chart(pfig,width='stretch')
    with right:
        st.write('**Electrical evidence**')
        st.write(f"Current/voltage slope: **{ev['current_voltage_slope']:.3f}** (ideal CC 0, CP -1, CZ +1)")
        st.write(f"Voltage variation available: **{'yes' if ev['slope_valid'] else 'no'}**")
        st.write(f"Normalised fundamental reactive power: **{ev['nominal_fundamental_reactive_fraction']:.3f}** (positive inductive, negative capacitive)")
        st.write(f"Frequency evidence available: **{'yes' if ev['spectral_valid'] else 'no'}**")
        st.write(f"Series-family fit residual: **{ev['spectral_fit_error']:.4f}**")
        if 'estimated_minimum_snr_db' in decision:st.write(f"Estimated signal quality: **{decision['estimated_minimum_snr_db']:.1f} dB SNR**")
        if pred.model.config.learned_physics_gate:
            st.write('Learned physics trust: '+', '.join(f"{k} {v:.1%}" for k,v in decision['physics_gates'].items()))
        st.caption('Component fractions are a rescaled equivalent fit, not measurements of hidden physical R/L/C values.')
    selected=decision.get('selected',[0])
    p=selected[-1]
    t=np.arange(waves.shape[-1])/sample_rate
    wf=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['Measured voltage','Measured current'])
    wf.add_trace(go.Scatter(x=t,y=waves[p,0],name='Voltage (V)',line_color='#087f82'),row=1,col=1)
    wf.add_trace(go.Scatter(x=t,y=waves[p,1],name='Current (A)',line_color='#eb9e31'),row=2,col=1)
    wf.update_layout(height=420,margin=dict(l=25,r=15,t=35,b=25));wf.update_xaxes(title_text='Time (s)',row=2,col=1)
    st.plotly_chart(wf,width='stretch')
    if 'trace' in result:
        st.write('**Measurement decisions**')
        st.dataframe(pd.DataFrame([{'observations':r['budget'],'selected probes':', '.join(PROBES[p] for p in r['selected']),
                          'confidence':r['confidence'],'accepted':r['accepted'],'estimated recording seconds (minimum)':r['recording_seconds_lower_bound']} for r in result['trace'][:decision['budget']]]),hide_index=True,width='stretch')
    if truth is not None and st.checkbox('Reveal simulator ground truth'):
        st.write(f"True labels: **{truth[0]} / {truth[1]}**. "+('Both predictions match.' if (BEHAVIOURS[decision['behaviour']],COMPOSITIONS[decision['composition']])==truth else 'At least one prediction is incorrect.'))
    if allow_export:
        st.download_button('Download measured CSV',export_csv(waves,frequencies,selected=selected,sample_rate=sample_rate),'ctee_measurements.csv','text/csv')

with demo:
    source=st.radio('Input source',['New simulated fixture','Held-out benchmark example','Measured CSV'],horizontal=True)
    if source=='New simulated fixture':
        controls=st.columns(4)
        composition=controls[0].selectbox('Passive branch',COMPOSITIONS,index=6)
        allowed=[b for b in BEHAVIOURS if b!='CP' or 'R' in composition]
        behaviour=controls[1].selectbox('Controller mode',allowed,index=len(allowed)-1)
        snr=controls[2].selectbox('Measurement noise',['Clean','40 dB','30 dB','20 dB','10 dB'],index=3)
        seed=controls[3].number_input('Fixture seed',min_value=0,max_value=100000,value=73,step=1)
        policy=st.selectbox('Measurement policy',['full','adaptive','fixed','random'])
        st.caption('Full uses all five probes for the main classifier result. Adaptive trades some accuracy for fewer measurements; fixed/random use confidence-based early stopping.')
        if st.button('Measure and classify',type='primary'):
            with threadpool_limits(limits=1):
                c=sample_circuit(composition,int(seed),np.random.default_rng(int(seed)))
                noise=None if snr=='Clean' else float(snr.split()[0])
                w,f=episode(c,behaviour,int(seed)+1,noise)
                _,sv,si=transient(c,sample_rate=5000);ss=np.stack([sv,si])*np.sqrt(2)
                ss[1]*=clean_probe(c,behaviour,0)[1]['gain'];ss=corrupt(ss,np.random.default_rng(int(seed)+2),noise)
                result=trace_policy(pred,w,f,ss,policy,int(seed))
            st.session_state['last']=(result,w,f,(behaviour,composition))
        if 'last' in st.session_state:show_prediction(*st.session_state['last'])
    elif source=='Held-out benchmark example':
        manifest=json.loads((RUN_ROOT/'data/manifest.json').read_text())
        name=st.selectbox('Unseen fixture recording',manifest['examples'])
        count=st.slider('Observations available',1,5,3)
        order=[0,1,3,2,4][:count]
        with np.load(RUN_ROOT/f'data/example_{name}.npz',allow_pickle=False) as item:
            w=item['waves'];f=item['frequencies'];ss=item['startup'];truth=(BEHAVIOURS[int(item['behaviour'])],COMPOSITIONS[int(item['composition'])])
        with threadpool_limits(limits=1):r=pred.predict(w,f,order,ss)
        r['selected']=order;show_prediction(r,w,f,truth)
    else:
        st.write('CSV columns: `probe,time,voltage,current,frequency`. Probe IDs: 0 nominal; 1 lower voltage; 2 higher voltage; 3 harmonic; 4 frequency probe. Use volts, amps, seconds, hertz. Startup capture is optional and is omitted for CSV input.')
        upload=st.file_uploader('Load aligned measured waveforms',type=['csv'])
        if upload:
            try:
                w,f,selected,sr=parse_csv(upload.getvalue())
                with threadpool_limits(limits=1):r=pred.predict(w,f,selected,sample_rate=sr)
                r['selected']=selected
                show_prediction(r,w,f,allow_export=False,sample_rate=sr)
                st.caption(f'Uploaded sample rate: {sr:.1f} Hz; source frequencies are read from the CSV metadata.')
            except (ValueError,UnicodeError) as e:st.error(str(e))

with results:
    path=RUN_ROOT/'reports/test_results.json'
    if path.exists():
        report=json.loads(path.read_text());rows=[]
        for name,conditions in report['conditions'].items():
            for condition,r in conditions.items():rows.append({'model':name,'condition':condition,'joint accuracy':r['joint_accuracy'],'behaviour macro-F1':r['behaviour_macro_f1'],'composition macro-F1':r['composition_macro_f1']})
        st.caption('Selected neural model: '+report['selected_neural']+' (validation selection). All figures below use held-out physical configurations.')
        st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        for filename in ['robustness.png','confusion_matrices.png','acquisition.png']:
            if (RUN_ROOT/'reports/figures'/filename).exists():st.image(str(RUN_ROOT/'reports/figures'/filename))
        with st.expander('Statistical comparison and limitations'):st.json(report['statistics'])
        if (RUN_ROOT/'reports/RESULTS.md').exists():st.markdown((RUN_ROOT/'reports/RESULTS.md').read_text())
    else:st.info('Final held-out comparisons will appear after the evaluation phase.')

with method:
    st.markdown('''### What the AI does
The network, backpropagation, optimizers, Fourier analysis and metrics are written explicitly using NumPy. It shares one learned electrical representation between two output heads. One predicts CC/CP/CZ behaviour; the other predicts the known passive branch class.

### Proposed contribution
Measured voltage-response and spectral fingerprints, quality-conditioned learned physics gates, averaged neural weights, valid electrical consistency, and uncertainty-guided acquisition are combined and compared with ordinary features, a noise-trained baseline, and analytical electrical rules. A new invention or a global best score is not established by a synthetic benchmark.

### Honest confidence
An accepted result needs multiple voltage levels, useful spectral evidence, a reasonable circuit-family fit, sufficient measured signal quality, and the calibrated confidence threshold. The noise-quality guard uses calibration devices only. Temperatures and acceptance thresholds are calibrated separately for each observation count. Outside the declared family, a confident answer can be wrong.
''')
    st.markdown((ROOT/'docs/LABEL_CONTRACT.md').read_text())


with parameters:
    st.subheader('The ML parameters: actual weights and biases')
    st.write('Every tensor below is learned by our own backpropagation and Adam/SGD updates. Initial values come from the recorded seed; the change columns show what training learned. Constant inputs can legitimately produce zero updates for some values.')
    summary=parameter_summary(pred.model)
    st.write(f"**{sum(r['count'] for r in summary):,} trainable values** across {len(summary)} tensors. `W0`, `W1` belong to the shared encoder; head tensors belong to behaviour/composition; gate tensors learn when to trust physics evidence.")
    st.dataframe(pd.DataFrame(summary),hide_index=True,width='stretch')
    st.download_button('Download every learned weight and bias',export_parameters(pred.model),'ctee_learned_parameters.csv','text/csv')
    tensor=st.selectbox('Weight or bias tensor',list(pred.model.params),key='parameter_tensor')
    matrix=pred.model.params[tensor]
    st.write('Selected tensor values')
    st.dataframe(pd.DataFrame(np.atleast_2d(matrix)),width='stretch')
    if st.checkbox('Show weight heatmap'):
        heatmap=go.Figure(go.Heatmap(z=np.atleast_2d(matrix),colorscale='RdBu',zmid=0))
        heatmap.update_layout(height=350,xaxis_title='Output coordinate',yaxis_title='Input coordinate')
        st.plotly_chart(heatmap,width='stretch')
    history_path=RUN_ROOT/'reports/training_history.json'
    learning_path=RUN_ROOT/'reports/parameter_learning.json'
    if learning_path.exists():
        learned=json.loads(learning_path.read_text())
        history=json.loads(history_path.read_text())[learned['selected_neural']]['history']
        st.write(f"Training used **{learned['optimizer_steps']:,} optimizer steps**; selected epoch **{learned['best_epoch']}** by validation loss; weight source **{learned.get('weight_source','raw')}**.")
        st.line_chart(pd.DataFrame(history).set_index('epoch')[['mean_gradient_norm','parameter_update_l2']])
        st.caption('Gradient norm measures the learning signal. Update norm measures the actual movement of all weights/biases in that epoch. Selected weights are restored from the best validation checkpoint.')
    st.subheader('Try changing one weight')
    st.write('This makes an isolated copy for an experiment. The deployed model and its saved benchmark stay intact. Manual edits require new evaluation and calibration before deployment.')
    row=int(st.number_input('Parameter row / bias index',min_value=0,max_value=matrix.shape[0]-1,value=0,step=1,key='parameter_row_'+tensor))
    index=(row,)
    if matrix.ndim==2:
        column=int(st.number_input('Parameter column',min_value=0,max_value=matrix.shape[1]-1,value=0,step=1,key='parameter_column_'+tensor))
        index=(row,column)
    delta=st.slider('Temporary weight change',-1.,1.,0.,step=.01,key='parameter_delta')
    st.caption(f'Current selected value: {matrix[index]:.8f}; experimental value: {matrix[index]+delta:.8f}')
    if st.button('Preview this weight change',key='preview_weight'):
        if 'last' not in st.session_state:st.info('First classify a new simulated fixture in the first tab.')
        else:
            last=st.session_state['last'][0];decision=last.get('decision',last)
            if 'features' not in decision:st.info('Classify the fixture again to capture the feature vector for this experiment.')
            else:
                with threadpool_limits(limits=1):_,edited=preview_edit(pred.model,np.array(decision['features']),tensor,index,float(delta))
                st.json(edited)
                st.caption('These are raw softmax probabilities before temperature calibration; compare original with edited, not with the calibrated headline confidence.')

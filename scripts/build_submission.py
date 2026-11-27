"""Build the academic report and shared chart data from the frozen experiment."""
import argparse, ast, copy, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/timestamped-report-20261127'
WORK=Path('/private/tmp/ctee-report-build')
FIG=WORK/'figures'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS={'w':W,'r':R,'a':'http://schemas.openxmlformats.org/drawingml/2006/main','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','pic':'http://schemas.openxmlformats.org/drawingml/2006/picture'}
def tag(x): return '{'+W+'}'+x
def load(name): return json.loads((ROOT/'runs/timestamped'/name).read_text())
def literal(path,name):
 for n in ast.parse((ROOT/path).read_text()).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets): return ast.literal_eval(n.value)
def clean(x): return json.loads(json.dumps(x,default=lambda v:v.item() if hasattr(v,'item') else str(v),allow_nan=False))
def records(df): return json.loads(df.to_json(orient='records',double_precision=15))

UNITS={'v_rms':'V','i_rms':'A','P_real_W':'W','S_apparent_VA':'VA','Q_nonactive_var':'var','pf':'dimensionless','i_crest_factor':'dimensionless','v_crest_factor':'dimensionless','i_form_factor':'dimensionless','i_skew':'dimensionless','i_kurtosis':'dimensionless','i_zero_cross':'crossings per window','f0_Hz':'Hz','cos_phi':'dimensionless','sin_phi':'dimensionless','i_h3_ratio':'ratio','i_h5_ratio':'ratio','i_h7_ratio':'ratio','i_thd':'ratio','v_thd':'ratio','i_spec_centroid_Hz':'Hz','i_spec_entropy':'extractor scale','i_hf_energy_ratio':'ratio'}
NAMES=['R','I','RC','RLC']
STAT={'p10':'10th percentile','median':'Median','p90':'90th percentile','iqr':'Interquartile range','early':'Median in first 20 percent of elapsed time','late':'Median in last 20 percent of elapsed time','slope':'Least squares slope per second','step':'Median absolute adjacent change per second'}

def prepare_data():
 OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
 data={'audit':json.loads((ROOT/'data/processed/audit.json').read_text()),'selection':load('selection.json'),'training':load('training.json'),'metrics':load('test_metrics.json')}
 data['comparison']=records(pd.read_csv(ROOT/'runs/timestamped/comparison.csv'))
 folds=pd.read_csv(ROOT/'runs/timestamped/cv_folds.csv').sort_values(['model','fold']).reset_index(drop=True)
 data['folds']=records(folds)
 data['predictions']=records(pd.read_csv(ROOT/'runs/timestamped/test_predictions.csv'))
 data['importance']=records(pd.read_csv(ROOT/'runs/timestamped/feature_importance.csv'))
 split=pd.read_csv(ROOT/'data/processed/splits.csv')
 data['counts']=[{'class':k,'development':int(((split.load==k)&(split.partition=='development')).sum()),'test':int(((split.load==k)&(split.partition=='test')).sum())} for k in NAMES]
 dictionary=[]
 for name in literal('load_classifier/data.py','MEASUREMENTS'):
  for stat,meaning in STAT.items(): dictionary.append({'feature':f'{name}__{stat}','measurement':name,'summary':stat,'meaning':meaning,'units':UNITS[name]+('/s' if stat in ['slope','step'] else '')})
 for row in data['importance']:
  if row['feature'].startswith('missingindicator_'): dictionary.append({'feature':row['feature'],'measurement':row['feature'].removeprefix('missingindicator_').split('__')[0],'summary':'missing flag','meaning':'1 when this summary is unavailable; 0 otherwise','units':'binary 0 or 1'})
 data['dictionary']=dictionary
 raw=pd.read_csv(ROOT/'examples/recordings.csv');mapping={'R':7,'I':6,'RC':1,'RLC':10}
 index=sorted(raw.window_idx.unique());wide=[]
 for idx in index:
  row={'elapsed_s':(idx+.5)/60}
  for label,recording in mapping.items():
   s=raw[(raw.id==recording)&(raw.window_idx==idx)]
   for field in ['i_rms','pf']: row[f'{label}_{field}']=None if s.empty else float(s.iloc[0][field])
  wide.append(row)
 data['time_windows']=wide;data['time_mapping']=mapping
 data['sources']=[['Run artifacts','runs/timestamped','Frozen comparison, scores, predictions and feature importances'],['Input metadata','data/meta_labeled.json','Supplied labels and household identifiers'],['Split and audit','data/processed/splits.csv and audit.json','Fixed houses, duplicate groups and source hashes'],['Time chart examples','examples/recordings.csv','Four development recordings; illustrative, not class averages'],['Original feature CSV','https://drive.google.com/file/d/1EihrOMdMkTDTsKAp37mTDrSCtwOvw5gG/view','209554 windows; SHA256 '+data['audit']['raw_csv_sha256']]]
 data['figure_map']=[['Figure 2.1','Recording counts','Overview','A16:D20'],['Figure 3.1','Current and power factor over elapsed time','Time patterns','A12:I'+str(12+len(wide))],['Figure 6.1','Classifier comparison','Model comparison','A5:F17 and CV folds A5:F53'],['Figure 6.2','Effect of the full feature representation','Feature study','A5:E9 and A13:C15'],['Figure 6.3','Test confusion matrix and per class scores','Test results','A5:F10 and A14:E18'],['Figure 7.1','Architecture','Not a numerical chart','Qualitative pipeline diagram'],['Figure 7.2','Training based feature importance','Feature importance','A5:B253']]
 (WORK/'report_data.json').write_text(json.dumps(data,indent=2,allow_nan=False))
 return data

def figures(d):
 import matplotlib
 matplotlib.use('Agg')
 global plt, FancyBboxPatch
 import matplotlib.pyplot as plt
 from matplotlib.patches import FancyBboxPatch
 plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 x=np.arange(4);fig,ax=plt.subplots(figsize=(7.5,3.2),layout='constrained')
 ax.bar(x-.19,[r['development'] for r in d['counts']],.38,label='Development',color='#34477a');ax.bar(x+.19,[r['test'] for r in d['counts']],.38,label='Test',color='#83a6c9');ax.set(xticks=x,xticklabels=NAMES,ylabel='Recording IDs',title='Recording counts by supplied load label');ax.legend();fig.savefig(FIG/'dataset_counts.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(7.5,4.8),layout='constrained',sharex=True)
 colors=['#34477a','#b25b28','#2b7a60','#8d619e']
 for label,c in zip(NAMES,colors):
  t=pd.DataFrame(d['time_windows']);axes[0].plot(t.elapsed_s,t[f'{label}_i_rms'],label=f'{label}  recording {d["time_mapping"][label]}',color=c);axes[1].plot(t.elapsed_s,t[f'{label}_pf'],color=c)
 axes[0].set(ylabel='RMS current (A)',title='Example development recordings over elapsed time');axes[0].legend(ncol=2,fontsize=8);axes[1].set(xlabel='Elapsed time (s)',ylabel='Power factor');fig.savefig(FIG/'time_patterns.png',dpi=200);plt.close(fig)
 comp=pd.DataFrame(d['comparison']).iloc[::-1]
 fig,ax=plt.subplots(figsize=(7.5,4.7),layout='constrained');ax.barh(comp.model,comp.macro_f1_mean,xerr=comp.macro_f1_std,capsize=3,color='#34477a');ax.set(xlim=(0,1.05),xlabel='Mean validation macro F1',title='Classifier comparison on development groups');ax.tick_params(axis='y',labelsize=9);fig.savefig(FIG/'model_comparison.png',dpi=220);plt.close(fig)
 m=d['metrics']['test'];matrix=np.array(m['confusion_matrix']);fig,axes=plt.subplots(2,1,figsize=(7.5,6.5),layout='constrained')
 axes[0].imshow(matrix,cmap='Blues',vmin=0);axes[0].set(xticks=range(4),yticks=range(4),xticklabels=NAMES,yticklabels=NAMES,xlabel='Predicted load',ylabel='Actual load',title='Reserved test confusion matrix')
 for (i,j),v in np.ndenumerate(matrix):axes[0].text(j,i,str(v),ha='center',va='center',color='white' if v>34 else 'black',fontsize=12)
 for key,offset,color in [('precision',-.24,'#34477a'),('recall',0,'#bf7f40'),('f1-score',.24,'#3a806a')]:axes[1].bar(np.arange(4)+offset,[m['per_class'][k][key] for k in NAMES],.24,label=key.replace('-score',''),color=color)
 axes[1].set(xticks=range(4),xticklabels=NAMES,ylim=(0,1.16),ylabel='Score',title='Per class test results');axes[1].legend(ncol=3,loc='upper center',fontsize=8);fig.savefig(FIG/'test_results.png',dpi=220);plt.close(fig)
 a=d['selection']['ablation'];fig,ax=plt.subplots(figsize=(7.5,3.2),layout='constrained');vals=[a['median_only_macro_f1_mean'],a['temporal_macro_f1_mean']];bars=ax.bar(['23 median summaries','184 full summaries'],vals,color=['#83a6c9','#34477a'],width=.55);ax.set(ylim=(0,1.04),ylabel='Mean validation macro F1',title='Feature representation comparison for selected classifier');ax.bar_label(bars,fmt='%.4f',padding=3);fig.savefig(FIG/'feature_study.png',dpi=200);plt.close(fig)
 top=pd.DataFrame(d['importance']).head(15).iloc[::-1];fig,ax=plt.subplots(figsize=(7.5,4.6),layout='constrained');ax.barh(top.feature,top.importance,color='#34477a');ax.set(xlabel='Impurity importance',title='Top 15 fitted forest feature importances');ax.tick_params(axis='y',labelsize=8);fig.savefig(FIG/'feature_importance.png',dpi=200);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7.5,4.2));ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
 boxes=[(0.4,5.6,4.2,1,'CSV feature windows\n23 selected measurements'),(5.4,5.6,4.2,1,'JSON metadata\nLabels and house groups'),(1.7,3.8,6.6,1,'Validate and order each recording\nCreate 184 summary features'),(1.7,2.1,6.6,1,'Training median imputation and 64 flags\n248 inputs to 400 Extra Trees'),(1.7,.4,6.6,1,'One predicted load label\nR  I  RC  RLC and model probabilities')]
 for x,y,w,h,text in boxes:
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='#eef2f8',edgecolor='#34477a',linewidth=1.2));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
 for start,end in [((2.5,5.6),(4,4.8)),((7.5,5.6),(6,4.8)),((5,3.8),(5,3.1)),((5,2.1),(5,1.4))]: ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#34477a','lw':1.3})
 fig.tight_layout();fig.savefig(FIG/'architecture.png',dpi=200);plt.close(fig)

def content(d):
 items=[]
 def h(text,level=1):items.append(('h'+str(level),text))
 def p(text):items.append(('p',text))
 def table(headers,rows,widths=None):items.append(('table',headers,rows,widths))
 def image(name,caption):items.append(('image',name,caption))
 def page():items.append(('page',))
 h('01 Introduction')
 h('1.1 Electrical loads and their measurement',2)
 p('Electrical equipment draws current in different ways. A heater, a fan and a laptop may operate from a similar supply voltage while producing different power factors, waveform shapes and harmonic content. These measurements can provide evidence for classifying the load category assigned to a recording. The present project studies this task using labelled electrical recordings supplied as JSON metadata and timestamped CSV feature windows.')
 p('We developed an offline supervised classifier for the four supplied labels R, I, RC and RLC. The selected Extra Trees model correctly classified 202 of 215 reserved test recordings, giving 93.95% accuracy and 0.9275 macro F1. These results describe agreement with the supplied annotations on houses excluded from training. They do not independently establish the physical internal circuit of an appliance.')
 h('1.2 Why classification matters',2)
 p('Load identification can support energy monitoring, equipment surveys and the interpretation of power-quality measurements. A useful classifier should distinguish its supported categories, expose the categories it confuses and provide a reproducible evaluation. These requirements matter more than a high training score, especially when neighbouring windows come from the same appliance recording.')
 p('Electrical signatures can overlap. Different devices can have similar power consumption, while the same device may change its current and power factor during starting and steady operation. This motivates a representation that includes both typical electrical values and changes over elapsed time. A prediction should also be interpreted within the data collection conditions for which it has been tested.')
 h('1.3 What supervised classification means',2)
 p('In supervised learning, each training example has an input and a known answer. Here, the input is a summary of one electrical recording, and the answer is one of the four load labels in the supplied JSON. The fitted model learns relationships between the measurements and those answers. This is a single multiclass task: each recording receives exactly one category. It is not reinforcement learning and does not involve rewards or an agent interacting with a power system.')
 p('Different classifier families learn different decision boundaries. Linear models are useful simple baselines, SVMs can learn curved boundaries, neural networks learn layered transformations, and tree ensembles combine numerical threshold decisions. A classifier should be chosen using data withheld from its fitting process. We therefore compared candidate settings using grouped cross-validation and selected the model with the strongest mean validation macro F1.')
 table(['Family','Decision approach','Role in this study'],[
 ['Logistic regression','Linear class boundaries after scaling','Simple interpretable comparison'],['SVM','Nonlinear radial basis kernel','Two regularization settings'],['Random forest','Average of fitted decision trees','Tree ensemble comparison'],['Extra Trees','Randomized numerical splits across many trees','Selected family and parameter variants'],['Gradient boosting','Sequential correction with shallow trees','Alternative nonlinear ensemble'],['Neural network','Dense hidden layers with nonlinear activations','MLP comparison with 64 and 32 hidden units']], [1.3,2.4,2.568])
 h('1.4 Why equations alone do not settle this dataset',2)
 p('Relations such as apparent power from RMS voltage and current remain useful electrical context. They do not, by themselves, uniquely recover every internal circuit from a short measured signature. Power electronics, distortion and different operating states can produce overlapping quantities. More fundamentally, this dataset supplies load annotations without an independent circuit-measurement record explaining how every annotation was established. Our model therefore learns the provided categories rather than replacing a physical circuit identification experiment.')
 p('The project does not claim a measured real-time deployment, automatic anomaly detection or simultaneous disaggregation of several appliances. Its working deliverable is a saved recording classifier, a CSV prediction command, the experiment results and an Excel workbook containing the chart data.')
 page();h('02 Objective and Electrical Context')
 h('2.1 The prediction target',2)
 p('The objective is to classify a complete recording as R, I, RC or RLC using the electrical features and their progression over elapsed time. The labels are retained exactly as supplied. In particular, I is not silently renamed to L. The JSON contains no CC, CP or CZ target, so the present experiment cannot train or evaluate the previous two-target task.')
 table(['Supplied label','Recording IDs','Annotation scope'],[[r['class'],r['development']+r['test'],'Provided load category'] for r in d['counts']],[1.2,1.3,3.768])
 image('dataset_counts.png','Figure 2.1 Recording IDs by supplied load label and partition. These counts include the retained duplicate IDs. Excel source: Overview A16:D20.')
 h('2.2 Behaviour classes and circuit labels',2)
 p('Constant-current, constant-power and constant-impedance behaviour describe how a load responds when supply voltage changes. Circuit labels describe a different aspect of electrical behaviour. Both are relevant background concepts, but a label for one cannot simply be assumed from a label for the other. The replacement dataset supplies one set of four load categories and no controlled voltage-sweep target for estimating CC, CP or CZ behaviour.')
 p('In ideal sinusoidal conditions, a resistor has current in phase with voltage, an inductor introduces lagging current and a capacitor introduces leading current. Combinations can overlap in their observed signatures. These textbook relationships help explain feature choices; they are not the independently verified labelling rules for this dataset.')
 h('2.3 Power factor phase and distortion',2)
 p('True power factor is real power divided by apparent power. The cosine of the fundamental phase angle represents displacement power factor. These quantities coincide under ideal sinusoidal conditions but may differ when current is distorted. We therefore use the supplied power factor, fundamental phase sine and cosine, and distortion measurements as separate inputs. Q_nonactive_var is treated as the supplied nonactive-power feature, not as a signed reactive-power label that uniquely determines topology.')
 h('2.4 Seasonal variation and timestamp scope',2)
 p('Seasonal and daily changes can alter the population and operating conditions seen by a deployed classifier. However, every metadata entry in this collection lists July 2013. The CSV timestamps describe elapsed seconds within individual recordings, rather than a calendar sequence across months or hours of a day. This dataset cannot support a measured seasonal curve, an intra-day demand profile or a claim of all-season accuracy. We therefore do not generate such curves or add month and hour predictors.')
 h('2.5 Power factor correction as a separate engineering task',2)
 p('Power-factor correction changes the reactive exchange between equipment and supply, for example through compensation hardware. Selecting compensation requires the applicable electrical assumptions, operating measurements and a specified target. It is separate from learning an appliance annotation. No capacitor-bank state or verified compensation target was supplied for these recordings.')
 p('We did not alter each window to force its power factor toward unity or 0.98. Doing so could remove phase and waveform information that the classifier needs and would not represent a measured corrected circuit. Any future compensation study should retain original measurements, document its assumptions and evaluate the resulting circuit separately. It must not be reported as a feature already used in this model.')
 h('2.6 Success criteria',2)
 p('The model must use timestamps meaningfully, exclude answer-bearing metadata from predictors, maintain separation between related recordings, compare suitable classifiers and export auditable test results. Accuracy is reported alongside per-class precision, recall and F1, balanced accuracy and a confusion matrix. Reproducibility is checked using saved source hashes, fixed splits and a repeated unchanged-seed experiment.')
 page();h('03 Dataset and Feature Design')
 h('3.1 Authoritative data sources',2)
 p('The authoritative inputs are meta_labeled.json and the combined all_features.csv from the supplied Drive folder. The JSON describes 1,074 IDs from 55 houses and eleven appliance types. The 214,166,054-byte CSV contains 209,554 rows and 64 columns. Each row is an already computed electrical-feature window. All recording IDs, load labels, house identifiers and appliance types match between the two sources.')
 p('The schema, 30 kHz sampling frequency, recording count and collection date are consistent with the original PLAID collection described by its authors [1]. This is a provenance inference about the supplied material. We did not substitute a separately downloaded PLAID version, and the added load annotations and original feature-extraction code have not been independently certified. The earlier simulated campus workbook is not used in this experiment.')
 h('3.2 Window and recording units',2)
 p('Each window covers 500 samples at 30,000 Hz, equivalent to one sixtieth of a second. end_sample is inclusive, whereas t_end_s is exclusive. Recordings contain 60 to 806 windows. We check that sample counts and elapsed time imply the stated sample rate, that windows are in order and do not overlap, and that the final times agree with the rounded lengths in the metadata. Each complete recording becomes one labelled model example.')
 image('time_patterns.png','Figure 3.1 Actual RMS current and supplied power factor for four development recordings. Each line is one illustrative recording, not a class mean or a test result. Elapsed time is not time of day. Excel source: Time patterns.')
 h('3.3 Selected original measurements',2)
 table(['Group','CSV measurement names','Count'],[
 ['Magnitude','v_rms, i_rms',2],['Power','P_real_W, S_apparent_VA, Q_nonactive_var, pf',4],['Phase','cos_phi, sin_phi',2],['Waveform shape','i_crest_factor, v_crest_factor, i_form_factor, i_skew, i_kurtosis, i_zero_cross',6],['Fundamental frequency','f0_Hz',1],['Harmonics and distortion','i_h3_ratio, i_h5_ratio, i_h7_ratio, i_thd, v_thd',5],['Spectral distribution','i_spec_centroid_Hz, i_spec_entropy, i_hf_energy_ratio',3],['Total selected measurements','',23]],[1.35,4.3,.618])
 p('These 23 columns already existed in the supplied CSV. Units include volts, amperes, watts, volt-amperes, var, hertz and dimensionless ratios; the workbook includes the measurement and summary dictionary. Spectral entropy retains the supplied extractor scale. Algebraically redundant quantities and equivalent L/C estimates are omitted. Recording ID, appliance name, load label, house, collection date, window count and duration are not direct predictors.')
 h('3.4 Newly computed recording features',2)
 p('For each selected measurement we compute eight summaries over the time-ordered windows. Percentiles and spread describe the distribution; early and late medians and rates describe temporal behaviour. The slope is a least-squares trend against actual elapsed seconds. The adjacent-change feature is the median absolute difference divided by the time between neighbouring valid windows; missing windows are not joined across a gap as if they were adjacent.')
 table(['Summary','Meaning'],[['p10','10th percentile, a representative low value'],['median','Middle value of the available measurements'],['p90','90th percentile, a representative high value'],['iqr','75th percentile minus 25th percentile'],['early','Median during the first 20 percent of elapsed time'],['late','Median during the last 20 percent of elapsed time'],['slope','Overall measurement trend per second'],['step','Typical absolute adjacent change per second']],[1.1,5.168])
 p('The feature count is 23 measurements multiplied by eight summaries, giving 184 engineered recording-level columns. Compared with a baseline containing one median for each measurement, the full representation adds 161 summary columns: seven additional summaries for each of the 23 measurements. These are transformations of existing measurements, not new sensor channels or synthetic recording examples.')
 h('3.5 Missing values and final input size',2)
 p('Some harmonic fields are missing in 0.94% of original windows. If a recording has no valid values for a summary, that summary stays missing until fitting. The preprocessing pipeline then replaces it with a median learned from training examples and records whether it was missing. In the final development fit, 64 such indicator columns are added. The fitted forest therefore receives 248 numeric inputs: 184 summaries plus 64 missingness flags. Indicator count is a property of the training data, not a new physical measurement count.')
 table(['Representation','Summary columns','Additional flags','Meaning'],[['Median baseline',23,'Fitted within each fold','One median per selected measurement'],['Full recording representation',184,64,'64 flags in the final development fit'],['Final selected forest input',248,'Included in total','184 summaries plus 64 flags']],[2.0,1.0,1.0,2.268])
 h('3.6 Duplicate checks and evaluation groups',2)
 p('No duplicate recording-window keys or infinite measurement values were found. Five pairs have identical required electrical and time columns: 234/389, 235/390, 236/391, 237/392 and 238/393. They connect house11 and house19 in development. We retain their IDs and place both houses in one validation group.')
 p('There are 1,069 distinct feature-summary rows across 1,074 supplied IDs. The 43 development houses form 42 connected groups. No identical summary crosses the development/test boundary. This grouping addresses known duplication but cannot establish physical appliance independence across every house because the metadata lacks a reliable physical appliance identifier.')
 h('3.7 Training validation and test protocol',2)
 p('A fixed outer split reserves 215 recordings from twelve houses for test and assigns 859 recordings from 43 houses to development. It is stratified approximately by load label and grouped by house, using seed 42. Within development, four validation folds use seed 43 and keep duplicate-connected houses together. Every candidate is fitted on three folds and scored on the fourth. Missing-value fitting and any scaling occur inside each training fold.')
 p('After an initial test evaluation, the duplicate connection was found and development folds were corrected. We repeated the same twelve candidates with unchanged test IDs. Corrected development scores selected the final model. The final result follows this data-integrity correction; test scores did not guide selection, but test data had previously been inspected.')
 page();h('04 Literature Survey')
 p('The following primary sources provide the dataset and algorithmic background. Their experiments use different datasets and tasks, so their scores are not treated as comparable benchmarks for our four supplied labels. The table is a focused methodological survey rather than an exhaustive review.')
 table(['Authors and source','Study','Contribution','Relevance and boundary'],[
 ['Medico and colleagues [1]','A voltage and current measurement dataset for plug load appliance identification in households, Scientific Data, 2020','Describes high-frequency domestic-appliance voltage and current recordings and associated metadata.','Supports dataset context. Our supplied engineered features and four added load labels still require their own provenance checks.'],
 ['Geurts, Ernst and Wehenkel [2]','Extremely randomized trees, Machine Learning, 2006','Introduces tree ensembles with randomized attribute and split-point choices.','Provides the basis for Extra Trees. It does not predict this project dataset score.'],
 ['Breiman [3]','Random Forests, Machine Learning, 2001','Develops randomized tree ensembles that combine multiple predictors.','Motivates a random-forest comparison, with our grouped protocol determining performance.'],
 ['Scikit-learn documentation [4,5]','ExtraTreesClassifier and StratifiedGroupKFold, version 1.8','Documents the implemented estimator and group-preserving validation splitter.','Grounds implementation and evaluation settings, rather than supplying a result for our labels.']],[1.2,1.65,1.65,1.768])
 p('The main lesson for the implementation is to separate a classifier method from the evidence supporting a particular application. A high reported score for appliance identity elsewhere does not establish an equivalent score for these load annotations, and a large number of windows does not imply a large number of independent devices.')
 page();h('05 Literature Summary and Research Gap')
 p('Our project addresses a practical evaluation need: build a working classifier for the supplied labelled recordings and make its data, feature construction and errors traceable. We do not claim to introduce a new classification algorithm or to establish a general gap in all electrical-load research.')
 table(['Issue to address','Response in this project'],[
 ['Electrical signatures overlap across categories','Use magnitude, phase, waveform and spectral measurements together.'],['Typical values hide changes within a recording','Add early, late, slope and adjacent-change summaries.'],['Related windows can inflate validation scores','Predict at recording level and separate houses.'],['Repeated records can cross house annotations','Connect houses sharing exact summaries before validation.'],['A high overall score can hide a weak class','Report all class precision, recall, F1 and the confusion matrix.'],['Figures alone do not expose their source numbers','Provide editable Excel charts and the associated data tables.'],['Physical annotation provenance is incomplete','State the limitation and recommend independent label verification.']],[2.5,3.768])
 p('The remaining scientific questions include whether I and RLC labels are physically consistent across devices, whether the same representation transfers to a new collection, and whether a more focused sequence representation improves development performance. These questions require additional evidence. They are not resolved by fitting the current training records perfectly.')
 page();h('06 Solution and Experimental Results')
 h('6.1 Selected classifier and comparison method',2)
 p('The solution is a supervised Extra Trees pipeline fitted to recording summaries. We tested a majority-class baseline, logistic regression, two SVM settings, random forest, gradient boosting, an MLP and several Extra Trees configurations. Three Extra Trees variants were added after an initial development comparison. The corrected comparison uses the same twelve candidates and ranks them by mean macro F1 across four validation folds.')
 p('The chosen setting, extra_trees_half, uses 400 trees, balanced class weights and max_features = 0.5. It achieved mean development accuracy 92.39% and mean macro F1 0.9056. Each class receives equal weight in macro F1. The forest is finally fitted on all 859 development recordings after model selection. No test recordings fit the estimator or its missing-value replacements.')
 table(['Candidate setting','Validation accuracy','Macro F1','F1 fold SD'],[[r['model'],f"{r['accuracy_mean']:.2%}",f"{r['macro_f1_mean']:.4f}",f"{r['macro_f1_std']:.4f}"] for r in d['comparison']],[2.5,1.25,1.25,1.268])
 image('model_comparison.png','Figure 6.1 Mean validation macro F1 for all twelve candidate settings. Error bars show the standard deviation across four folds, not a confidence interval. Excel sources: Model comparison and CV folds.')
 h('6.2 Effect of the additional features',2)
 a=d['selection']['ablation'];gain=a['temporal_macro_f1_mean']-a['median_only_macro_f1_mean']
 p(f"For the selected classifier configuration, the median-only representation achieved mean validation macro F1 {a['median_only_macro_f1_mean']:.4f}. The full representation achieved {a['temporal_macro_f1_mean']:.4f}, an absolute increase of {gain:.4f} F1, or {gain*100:.2f} points on a 0–100 scale. This is a development result, not an improvement measured by comparing two test scores.")
 p('The comparison adds 161 summary columns beyond the 23-median baseline and changes the training-observed missingness indicators. It adds distribution and time summaries together. We can therefore attribute the result to the full representation as a whole, but cannot isolate a benefit caused solely by timestamps or by any one new feature. No separate median-only test score was measured. The full representation improved one validation fold and was slightly lower in the other three; the higher mean does not establish a consistent or statistically significant benefit.')
 image('feature_study.png','Figure 6.2 Validation feature study for the selected Extra Trees configuration. Both versions use the same grouped folds and training-only imputation. Excel source: Feature study.')
 h('6.3 Final reserved test results',2)
 m=d['metrics']['test']
 table(['Measure','Result'],[['Correct recordings','202 of 215'],['Accuracy','93.95%'],['Macro F1','0.9275'],['Balanced accuracy','92.50%'],['Weighted F1','0.9397'],['Majority-class baseline accuracy','31.63%']],[3.5,2.768])
 p('Accuracy is the fraction of correct recording predictions. Balanced accuracy averages the four recalls. Macro F1 averages class F1 scores equally; weighted F1 weights them by their test support. The majority baseline always predicts RC, the most common development label, and correctly classifies only 68 test recordings.')
 table(['Class','Precision','Recall','F1','Support'],[[k,f"{m['per_class'][k]['precision']:.4f}",f"{m['per_class'][k]['recall']:.4f}",f"{m['per_class'][k]['f1-score']:.4f}",int(m['per_class'][k]['support'])] for k in NAMES],[1.0,1.3,1.3,1.3,1.368])
 image('test_results.png','Figure 6.3 Reserved-test confusion matrix and per-class scores. Rows are actual labels and columns are predictions. All 215 test recordings come from houses absent from development. Excel source: Test results, calculated from Test predictions.')
 h('6.4 Errors and uncertainty',2)
 p('All thirteen test mistakes are false RLC predictions: ten actual I recordings and three actual R recordings. Seven Fan recordings and three Fridge recordings account for the I errors; three Heater recordings account for the R errors. I recall is 75%, while RLC precision is 78.33%. Finding all actual RLC recordings therefore does not mean every RLC prediction is correct.')
 p('A whole-house bootstrap with 2,000 resamples gives an approximate 95% accuracy interval of 89.30% to 97.77% and macro F1 interval of 0.8502 to 0.9730. Resampling entire houses keeps related recordings together. These intervals are conditional on this fitted model and the observed test houses; they do not include every source of label, model-selection or population uncertainty.')
 p('Training accuracy is 100%, which indicates a close fit to the development recordings, not a guarantee for new data. The validation and test results are the relevant generalization evidence. The test set contains all four load classes but only ten of eleven appliance types; Washing Machine has no test recordings. We cannot claim type-specific test performance for that category.')
 page();h('07 Architecture and Reproducibility')
 h('7.1 Architecture at a glance',2)
 image('architecture.png','Figure 7.1 The implemented recording-classification pipeline. Metadata provides labels and evaluation groups; only electrical/time-derived features enter the fitted classifier. This is a qualitative diagram, not a numerical chart.')
 p('The implementation is a one-way pipeline. It validates and matches recording data, summarizes the ordered windows, fits median replacements and missing-value indicators on training examples, applies the forest and exports the predicted load with four probability values. It produces a single load answer per complete recording. There is no shared neural backbone or second prediction head in the selected artifact.')
 h('7.2 How the forest works',2)
 p('An individual tree partitions training examples through numerical threshold questions. Extra Trees proposes randomized thresholds and evaluates splits using the training labels. In the selected configuration, a randomly selected half of the available columns is considered at each split. With 248 fitted numeric columns, that corresponds to 124 candidate columns. Averaging 400 randomized trees reduces dependence on one particular tree [2,4].')
 table(['Setting','Selected value'],[['Estimator','ExtraTreesClassifier from scikit-learn 1.8.0'],['Number of trees','400'],['Candidate features per split','0.5 of available columns'],['Class weights','balanced'],['Minimum samples in a leaf','1'],['Minimum samples to split','2'],['Maximum depth','No fixed maximum'],['Bootstrap sampling','Disabled'],['Random seed','42'],['Feature scaling','Not required by selected forest']],[2.5,3.768])
 p('Balanced class weights are computed from development label frequencies so rarer classes have greater per-example weight. The selected forest uses Gini-based split quality. It is not trained with Adam, back-propagation, dropout or epochs. Those terms apply to the separate neural-network candidate, which had hidden layers of 64 and 32 units and lower mean validation performance in this experiment.')
 h('7.3 Data processing and model persistence',2)
 p('The measurement allowlist prevents labels or additional metadata columns from becoming predictors. Source missing values remain explicit until the training imputer is fitted. Trees do not require the standard scaling used for the logistic, SVM and MLP comparisons. A saved joblib bundle contains the fitted preprocessing and classifier, class order, feature schema and source fingerprints. The same feature construction and saved preprocessing are used at inference.')
 p('The command-line sequence is prepare, compare, train, evaluate and verify. A predict command accepts a feature CSV with complete recording windows and writes one row per ID. The row includes predicted_load and probabilities for R, I, RC and RLC. Example recordings are provided without labels or appliance names. The output probabilities have not been independently calibrated and should not be interpreted as guaranteed correctness percentages.')
 h('7.4 Delivery layer and possible API extension',2)
 p('The current working interface is a local CSV command and a saved Python model. No FastAPI service, browser dashboard, live meter connection or cloud deployment was implemented for this version. An API can be added later by validating the same schema, collecting a complete recording and calling the existing prediction pipeline. That extension should include latency measurements and deployment-specific testing rather than inheriting an unsupported real-time claim.')
 h('7.5 Feature importance and explanation scope',2)
 p('We export the fitted forest’s impurity-based feature importances for all 248 transformed columns. They describe how split gains are distributed during fitting. Correlated inputs can share importance or distort its interpretation, so these values are not causal explanations and do not reconstruct individual decisions. They should not be called SHAP values. SHAP, calibration and an abstention policy are possible future additions and have not been evaluated here.')
 image('feature_importance.png','Figure 7.2 Top fifteen training-based forest feature importances. These are impurity importances, not SHAP attributions or evidence of physical causation. Excel source: Feature importance; all 248 values are retained.')
 h('7.6 Components and chart data export',2)
 table(['Component','Implementation','Output'],[['Input validation','data.py','Checked windows and labelled recording IDs'],['Feature construction','features.py','184 ordered-recording summaries'],['Model definitions','models.py','Candidate preprocessing and classifiers'],['Experiment','experiment.py','Grouped comparisons and fitted winner'],['Evaluation','evaluate.py','Test metrics, figures and verification'],['Inference','predict.py','One label and four probabilities per recording'],['Reporting','Submission builders','Word report and Excel chart data']],[1.4,1.7,3.168])
 p('The companion chart_data.xlsx supplies all numerical figure data, validation folds and test predictions, plus a feature dictionary and figure map. Native charts reference worksheet cells; confusion counts and metrics use Excel formulas so the results and figures can be recreated.')
 h('7.7 Verification and reproducibility',2)
 p('Twenty tests passed locally, in a clean checkout and in GitHub Actions. Saved inference and an independent recount reproduce all test results. Data preparation reproduces the saved files byte for byte. Unchanged-seed retraining reproduces selection, 400 trees and 1,074 class predictions, with probabilities within 1e-14. This checks local reproducibility, not multi-seed or external robustness.')
 page();h('08 Conclusion and Further Work')
 h('8.1 Project achievements',2)
 p('We built and verified a supervised recording classifier for the supplied R, I, RC and RLC labels. The project uses 23 existing electrical measurements to construct 184 recording summaries. Relative to a 23-median baseline, the full representation adds 161 summary columns; the final imputer adds 64 missingness indicators, giving 248 forest inputs. The selected 400-tree Extra Trees configuration achieves 93.95% accuracy on 215 reserved recordings from twelve houses.')
 p('The full representation improved mean development macro F1 from 0.8899 to 0.9056 for the selected classifier setting. The final test macro F1 is 0.9275. These are separate validation and test observations. They support this implementation for the supplied annotation task, while I recall of 75% shows a clear remaining weakness. The Excel workbook makes the figures auditable and allows their data to be reused in academic presentation.')
 h('8.2 Priorities for improvement',2)
 p('The first priority is to inspect I/RLC annotation consistency and obtain more independent Fan, Fridge and mixed-load recordings across houses. The original waveform files and extraction code would allow electrical-feature calculations and low-signal cases to be checked. Reliable physical appliance IDs would strengthen grouping beyond household metadata.')
 p('Further development can compare selected feature groups, alternate time summaries, controlled sequence models and additional classifier settings using development groups only. A future study should obtain a fresh external test collection, include the missing appliance types and cover a wider range of operating conditions. Calibration, SHAP explanations, API serving, streaming inference and any power-factor correction experiment should each be implemented and tested as separate extensions.')
 page();h('09 References')
 refs=[
 '[1] Medico, R., De Baets, L., Gao, J., Giri, S., Kara, E., Dhaene, T., Develder, C., Bergés, M., and Deschrijver, D. (2020). A voltage and current measurement dataset for plug load appliance identification in households. Scientific Data 7, 49. https://doi.org/10.1038/s41597-020-0389-7',
 '[2] Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. Machine Learning 63, 3–42. https://doi.org/10.1007/s10994-006-6226-1',
 '[3] Breiman, L. (2001). Random Forests. Machine Learning 45, 5–32. https://doi.org/10.1023/A:1010933404324',
 '[4] Scikit-learn documentation, version 1.8. ExtraTreesClassifier. https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html',
 '[5] Scikit-learn documentation, version 1.8. StratifiedGroupKFold. https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html',
 '[6] Supplied project data. meta_labeled.json and all_features.csv, Drive folder https://drive.google.com/drive/folders/1_aO3gFYwigITWH0YmCPAoB48wRg2B5mF. The CSV SHA256 is 5fee755b96b19de6cec432896828ac24cbcaf7866bde3df73f9551c2e24f69a4.',
 '[7] Project implementation and experiment artifacts. CTEE repository, timestamped-load-classifier branch, commit 21e74ee. https://github.com/shantanu-1607/CTEE/tree/timestamped-load-classifier. Source artifacts: data/processed/audit.json and runs/timestamped comparison, selection, training, test metrics, predictions and verification files.',
 '[8] Companion deliverable. chart_data.xlsx, prepared with this report from the frozen experiment artifacts. The Overview worksheet maps each numerical figure to its data range. External references accessed 27 November 2026.'
 ]
 for ref in refs:p(ref)
 return items

def build_docx(reference,d):
 global E
 from lxml import etree as E
 original=reference.read_bytes();baseline_hash=hashlib.sha256(original).hexdigest()
 archive=zipfile.ZipFile(reference);parts={n:archive.read(n) for n in archive.namelist()}
 xml=E.fromstring(parts['word/document.xml']);body=xml.find(tag('body'))
 ps=body.findall(tag('p'));source_table=body.find(tag('tbl'))
 paragraph_templates={'p':copy.deepcopy(ps[23]),'h1':copy.deepcopy(ps[21]),'h2':copy.deepcopy(ps[22]),'caption':copy.deepcopy(ps[73]),'cover':copy.deepcopy(ps[1]),'name':copy.deepcopy(ps[6])}
 pictemplate=next(copy.deepcopy(p) for p in ps if p.xpath('.//w:drawing',namespaces=NS))
 sect=copy.deepcopy(body.find(tag('sectPr')))
 for child in list(body):body.remove(child)
 def para(text,role='p',page=False):
  node=copy.deepcopy(paragraph_templates[role]);pr=node.find(tag('pPr'));rp=next(iter(node.xpath('./w:r/w:rPr',namespaces=NS)),None);rp=copy.deepcopy(rp) if rp is not None else E.Element(tag('rPr'))
  for ch in list(node):
   if ch.tag!=tag('pPr'):node.remove(ch)
  if pr is None:pr=E.SubElement(node,tag('pPr'))
  # Discard source bookmark/revision identity while retaining paragraph typography.
  node.attrib.clear()
  if page:E.SubElement(pr,tag('pageBreakBefore'))
  for j,line in enumerate(str(text).split('\n')):
   r=E.SubElement(node,tag('r'));r.append(copy.deepcopy(rp))
   if j:E.SubElement(r,tag('br'))
   t=E.SubElement(r,tag('t'));t.text=line;t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
  body.append(node);return node
 def table(headers,rows,widths):
  tbl=E.Element(tag('tbl'));pr=copy.deepcopy(source_table.find(tag('tblPr')));tbl.append(pr)
  pr.find(tag('tblW')).set(tag('w'),'9026')
  widths=widths or [6.268/len(headers)]*len(headers);twips=[round(x*1440) for x in widths];twips[-1]+=9026-sum(twips)
  grid=E.SubElement(tbl,tag('tblGrid'))
  for width in twips:E.SubElement(grid,tag('gridCol'),{tag('w'):str(width)})
  templates=source_table.findall(tag('tr'))[:2]
  for i,values in enumerate([headers,*rows]):
   tr=E.SubElement(tbl,tag('tr'));trpr=E.SubElement(tr,tag('trPr'));E.SubElement(trpr,tag('cantSplit'))
   if i==0:E.SubElement(trpr,tag('tblHeader'))
   for j,value in enumerate(values):
    sourcecell=templates[0 if i==0 else 1].find(tag('tc'));tc=E.SubElement(tr,tag('tc'));tcpr=copy.deepcopy(sourcecell.find(tag('tcPr')));tc.append(tcpr)
    size=tcpr.find(tag('tcW'))
    if size is None:size=E.SubElement(tcpr,tag('tcW'))
    size.set(tag('w'),str(twips[j]));size.set(tag('type'),'dxa')
    margins=tcpr.find(tag('tcMar'))
    if margins is None:margins=E.SubElement(tcpr,tag('tcMar'))
    for side in ['top','bottom','left','right']:
     el=margins.find(tag(side))
     if el is None:el=E.SubElement(margins,tag(side))
     el.set(tag('w'),'75');el.set(tag('type'),'dxa')
    p=copy.deepcopy(sourcecell.find(tag('p')));ppr=p.find(tag('pPr'))
    if ppr is None:ppr=E.SubElement(p,tag('pPr'))
    for child in list(p):
     if child.tag!=tag('pPr'):p.remove(child)
    jc=ppr.find(tag('jc'))
    if jc is None:jc=E.SubElement(ppr,tag('jc'))
    jc.set(tag('val'),'center' if i==0 or isinstance(value,(int,float)) or (j>0 and len(str(value))<12) else 'left')
    spacing=ppr.find(tag('spacing'))
    if spacing is None:spacing=E.SubElement(ppr,tag('spacing'))
    spacing.set(tag('after'),'25');spacing.set(tag('line'),'240');spacing.set(tag('lineRule'),'auto')
    r=E.SubElement(p,tag('r'));rp=E.SubElement(r,tag('rPr'));E.SubElement(rp,tag('rFonts'),{tag('ascii'):'Arial',tag('hAnsi'):'Arial'});E.SubElement(rp,tag('sz'),{tag('val'):'20'});E.SubElement(rp,tag('color'),{tag('val'):'FFFFFF' if i==0 else '000000'})
    if i==0:E.SubElement(rp,tag('b'))
    E.SubElement(r,tag('t')).text=str(value);tc.append(p)
  body.append(tbl);para('')
 # Cover preserves source centered title and metadata roles without unconfirmed identities.
 title=para('Electrical Load Classification\nUsing Timestamped Measurements','cover');pr=title.find(tag('pPr'));E.SubElement(pr,tag('pStyle'),{tag('val'):'Title'})
 pr.find(tag('spacing')).set(tag('before'),'2200')
 para('PROJECT REPORT','name');para('PRESENTED BY','cover')
 # Smaller metadata role follows the source cover; title typography is confined to title.
 body[-1].find('.//'+tag('sz')).set(tag('val'),'24')
 para('SHANTANU SINGH\nBTECH/10232/24','name');para('AANYAA SINHA\nBTECH/10789/24','name');para('27 November 2026','name')
 para('Contents','h1',page=True)
 chapters=[x[1] for x in content(d) if x[0]=='h1']
 for chapter in chapters:
  n=para(chapter);n.find(tag('pPr')).find(tag('jc')).set(tag('val'),'left')
 rels=E.fromstring(parts['word/_rels/document.xml.rels']);REL='http://schemas.openxmlformats.org/package/2006/relationships'
 oldmedia=[]
 for rel in list(rels):
  if rel.get('Type','').endswith('/image'):
   oldmedia.append('word/'+rel.get('Target'));rels.remove(rel)
 for name in oldmedia:parts.pop(name,None)
 media={};picture_number=100;nextpage=True
 for item in content(d):
  kind=item[0]
  if kind=='page':nextpage=True;continue
  if kind in ['h1','h2','p']:
   para(item[1],kind if kind!='p' else 'p',page=(kind=='h1' and nextpage));nextpage=False
  elif kind=='table':table(*item[1:])
  elif kind=='image':
   from PIL import Image
   name,caption=item[1:];picture_number+=1;rid=f'rId{picture_number}'
   path=FIG/name;media['word/media/'+name]=path.read_bytes();E.SubElement(rels,'{'+REL+'}Relationship',Id=rid,Type=R+'/image',Target='media/'+name)
   node=copy.deepcopy(pictemplate);node.attrib.clear();ppr=node.find(tag('pPr'))
   if ppr is None:ppr=E.SubElement(node,tag('pPr'))
   E.SubElement(ppr,tag('keepNext'));jc=ppr.find(tag('jc'))
   if jc is None:jc=E.SubElement(ppr,tag('jc'))
   jc.set(tag('val'),'center')
   for old in node.xpath('./w:r[not(w:drawing)]',namespaces=NS):node.remove(old)
   for blip in node.xpath('.//a:blip',namespaces=NS):blip.set('{'+R+'}embed',rid)
   w,h=Image.open(path).size;cx=round(6.05*914400);cy=round(cx*h/w)
   for ext in node.xpath('.//wp:extent | .//a:xfrm/a:ext',namespaces=NS):ext.set('cx',str(cx));ext.set('cy',str(cy))
   for prop in node.xpath('.//wp:docPr | .//pic:cNvPr',namespaces=NS):prop.set('id',str(picture_number));prop.set('name',name);prop.set('descr',caption)
   body.append(node);para(caption,'caption')
 body.append(sect)
 parts['word/document.xml']=E.tostring(xml,xml_declaration=True,encoding='UTF-8',standalone=True)
 parts['word/_rels/document.xml.rels']=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
 # Replace provenance metadata from the reference with the report's confirmed authors.
 if 'docProps/core.xml' in parts:
  core=E.fromstring(parts['docProps/core.xml']);dc='http://purl.org/dc/elements/1.1/';cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties'
  for field,value in [(dc+'title','Electrical Load Classification Using Timestamped Measurements'),(dc+'creator','Shantanu Singh and Aanyaa Sinha'),(cp+'lastModifiedBy','Shantanu Singh and Aanyaa Sinha')]:
   e=core.find('{'+field+'}')
   if e is None:e=E.SubElement(core,'{'+field+'}')
   e.text=value
  parts['docProps/core.xml']=E.tostring(core,xml_declaration=True,encoding='UTF-8',standalone=True)
 target=OUT/'electrical_load_classification_report.docx'
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for n,b in {**parts,**media}.items():z.writestr(n,b)
 preserved=[n for n in archive.namelist() if n not in ['word/document.xml','word/_rels/document.xml.rels','docProps/core.xml'] and not n.startswith('word/media/')]
 with zipfile.ZipFile(target) as z:
  assert all(archive.read(n)==z.read(n) for n in preserved)
 assert hashlib.sha256(reference.read_bytes()).hexdigest()==baseline_hash
 (WORK/'document-integrity.json').write_text(json.dumps({'reference_sha256':baseline_hash,'preserved_parts':preserved,'preserved_parts_identical':True,'output':str(target)},indent=2))
 print('Created',target)

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference',type=Path);parser.add_argument('--figures-only',action='store_true');args=parser.parse_args()
 if args.figures_only:
  figures(json.loads((WORK/'report_data.json').read_text()))
 else:
  if not args.reference:parser.error('--reference is required')
  import subprocess
  d=prepare_data()
  subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--figures-only'],check=True)
  build_docx(args.reference,d)

import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gunzipSync} from 'node:zlib';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const root=path.dirname(fileURLToPath(import.meta.url));
const d=JSON.parse(gunzipSync(await fs.readFile(path.join(root,'results/workbook_data.json.gz'))));
const wb=Workbook.create();
const names=['Overview','Future monthly','Future hourly','Hourly predictions','Monthly predictions','Monthly trends','Seasonal errors','Validation','Example week','Data quality'];
const sheets=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const navy='#284B7D',orange='#CF7C35',green='#418474';
function base(name,rows,cols){const s=sheets[name];s.showGridLines=false;s.getRangeByIndexes(0,0,Math.min(rows,200),cols).format.font={name:'Arial',size:10,color:'#182838'};s.getRangeByIndexes(0,0,Math.min(rows,200),cols).format.rowHeight=22;s.getRangeByIndexes(0,0,Math.min(rows,200),cols).format.columnWidth=18;s.getRange('A2').values=[[name]];s.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:navy};return s;}
function table(s,row,col,headers,rows){s.getRangeByIndexes(row-1,col-1,rows.length+1,headers.length).values=[headers,...rows];s.getRangeByIndexes(row-1,col-1,1,headers.length).format={fill:navy,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:38,horizontalAlignment:'center',verticalAlignment:'center'};}
function note(s,cell,text){s.getRange(cell).values=[[text]];s.getRange(cell).format.font={name:'Arial',size:10,italic:true,color:'#465568'};}
function chart(s,type,range,title,from,to,percent=false){const c=s.charts.add(type,Array.isArray(range)?range.map(r=>s.getRange(r)):s.getRange(range));c.title=title;c.titleTextStyle.fontSize=13;c.titleTextStyle.typeface='Arial';c.setPosition(from,to);c.legend={position:'top',textStyle:{typeface:'Arial',fontSize:10}};c.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:10}};c.yAxis={min:0,numberFormatCode:percent?'0%':'0.0',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:10}};if(type==='bar'){c.barOptions.direction='column';c.barOptions.grouping='clustered';}c.series.items.forEach((x,i)=>{x.fill=[navy,orange,green][i%3];if(type==='line')x.line={fill:[navy,orange,green][i%3],style:'solid',width:1.7};});return c;}
// Excel has no timezone. Preserve Asia/Kolkata wall-clock values as serial dates.
const date=x=>new Date(x.endsWith('Z')?x:x+'Z');
const hn=d.hourly_predictions.length+5,mn=d.monthly_predictions.length+5;
let s=base('Hourly predictions',hn,10);
table(s,5,1,['Local timestamp','Building','Actual kW','Forecast kW','Last-week kW','Peak cutoff kW','Actual peak 1/0','Forecast peak 1/0','Absolute error kW','Baseline error kW'],d.hourly_predictions.map(r=>[date(r.timestamp),r.building,r.actual_kw,r.predicted_kw,r.baseline_kw,r.peak_threshold_kw,r.actual_peak,r.predicted_peak,null,null]));
s.getRange('A:A').format.columnWidth=25;s.getRange('B:B').format.columnWidth=20;s.getRange(`A6:A${hn}`).setNumberFormat('yyyy-mm-dd hh:mm');s.getRange(`C6:F${hn}`).setNumberFormat('0.000');s.getRange(`I6:J${hn}`).setNumberFormat('0.000');
s.getRange(`I6:J${hn}`).formulas=d.hourly_predictions.map((_,i)=>[`=ABS(C${i+6}-D${i+6})`,`=ABS(C${i+6}-E${i+6})`]);s.freezePanes.freezeRows(5);
s.getRange(`G6:H${hn}`).formulas=d.hourly_predictions.map((_,i)=>[`=IF(C${i+6}>F${i+6},1,0)`,`=IF(D${i+6}>F${i+6},1,0)`]);
note(s,'A3','Reserved 2017 hourly targets. Peak cutoff was fixed from pre-2017 training data.');
s=base('Monthly predictions',mn,9);
table(s,5,1,['Forecast origin','Target month','Building','Lead months','Actual mean kW','Forecast mean kW','Last-year mean kW','Absolute error kW','Baseline error kW'],d.monthly_predictions.map(r=>[date(r.origin),r.target_month,r.building,r.lead_months,r.actual_kw,r.predicted_kw,r.baseline_kw,null,null]));
s.getRange('A:A').format.columnWidth=21;s.getRange('C:C').format.columnWidth=21;s.getRange(`A6:A${mn}`).setNumberFormat('yyyy-mm-dd');s.getRange(`E6:I${mn}`).setNumberFormat('0.000');s.getRange(`H6:I${mn}`).formulas=d.monthly_predictions.map((_,i)=>[`=ABS(E${i+6}-F${i+6})`,`=ABS(E${i+6}-G${i+6})`]);s.freezePanes.freezeRows(5);
note(s,'A3','One record per origin/building/horizon; repeated target months are not independent.');
s=base('Future monthly',30,15);
table(s,5,1,['Origin','Building','Target month','Lead months','Forecast mean kW'],d.future_monthly.map(r=>[date(r.origin),r.building,r.target_month,r.lead_months,r.forecast_kw]));
s.getRange('A:A').format.columnWidth=20;s.getRange('B:B').format.columnWidth=20;s.getRange('A6:A14').setNumberFormat('yyyy-mm-dd');s.getRange('E6:E14').setNumberFormat('0.00');
table(s,5,7,['Month','Academic','Facilities','Mess'],['2018-01','2018-02','2018-03'].map(m=>[m,null,null,null]));
for(let i=0;i<3;i++)for(let j=0;j<3;j++){const col=String.fromCharCode(72+j),row=6+i;s.getRange(`${col}${row}`).formulas=[[`=SUMIFS($E$6:$E$14,$C$6:$C$14,$G${row},$B$6:$B$14,${col}$5)`]];}
s.getRange('H6:J8').setNumberFormat('0.00');chart(s,'line','G5:J8','Historical-origin monthly forecast (kW)','G11','O27');
note(s,'A3','Issued at 1 January 2018 from data through 2017. These are NOT forecasts for 2026.');note(s,'A29','Future actuals were not supplied; no accuracy is claimed for these nine forecasts.');
s=base('Future hourly',80,16);
table(s,5,1,['Local target hour','Building','Forecast kW','Peak cutoff kW','Peak 1 / non-peak 0'],d.future_hourly.map(r=>[date(r.timestamp),r.building,r.forecast_kw,r.peak_threshold_kw,r.predicted_peak]));
s.getRange('A:A').format.columnWidth=25;s.getRange('B:B').format.columnWidth=20;s.getRange('A6:A77').setNumberFormat('yyyy-mm-dd hh:mm');s.getRange('C6:D77').setNumberFormat('0.00');s.freezePanes.freezeRows(5);
table(s,5,7,['Local hour','Academic kW','Cutoff kW'],Array.from({length:24},(_,i)=>[`${String(i).padStart(2,'0')}:00`,null,null]));
s.getRange('H6:I29').formulas=Array.from({length:24},(_,i)=>[`=C${i+6}`,`=D${i+6}`]);s.getRange('H6:I29').setNumberFormat('0.00');const fc=chart(s,'line','G5:I29','Academic forecast for 1 January 2018','G32','P50');fc.xAxis={axisType:'textAxis',tickLabelInterval:3};
note(s,'A3','Operational refit through 2017; peak cutoffs differ from frozen evaluation cutoffs.');
s=base('Monthly trends',60,15);
for(let k=0;k<3;k++){
 const building=['Academic','Facilities','Mess'][k],start=5+k*18;
 const months=Array.from({length:12},(_,i)=>`2017-${String(i+1).padStart(2,'0')}`);
 const rows=months.map(m=>{const r=d.monthly_chart.find(r=>r.building===building&&r.target_month===m);return [m,r?.actual_kw??null,r?.predicted_kw??null,r?.baseline_kw??null];});
 table(s,start,1,['Month',`${building} actual kW`,'1-month forecast kW','Last-year kW'],rows);
 s.getRangeByIndexes(start,1,12,3).setNumberFormat('0.00');chart(s,'line',`A${start}:D${start+12}`,`${building}: monthly mean demand (kW)`,`F${start}`,`O${start+15}`);
}
note(s,'A3','All eligible one-month-ahead 2017 forecasts. Low-coverage months stay blank.');s.getRange('B:D').format.columnWidth=24;
s=base('Seasonal errors',27,15);
table(s,5,1,['Season group','Hourly WAPE','Monthly WAPE','Hourly targets','Monthly records'],d.seasons.map(r=>[r.season,r.hourly_wape,r.monthly_wape,r.hourly_n,r.monthly_n]));s.getRange('B6:C9').setNumberFormat('0.00%');chart(s,'bar','A5:C9','Seasonal forecast errors — lower is better','G5','O24',true);
note(s,'A3','Month groups are explicit conventions. WAPE is error, not accuracy.');
s=base('Validation',38,23);
table(s,5,1,['Task','Fold','Candidate','Building WAPE','MAE kW','RMSE kW','Overall WAPE','R squared','Records','Seconds'],d.cv.map(r=>[r.task,r.fold,r.model,r.macro_building_wape,r.mae_kw,r.rmse_kw,r.wape,r.r2,r.n,r.seconds]));s.getRange('C:C').format.columnWidth=30;s.getRange('D6:D29').setNumberFormat('0.00%');s.getRange('G6:G29').setNumberFormat('0.00%');s.getRange('E6:F29').setNumberFormat('0.000');s.freezePanes.freezeRows(5);
const pairs=[...new Set(d.cv.map(r=>r.task+' / '+r.model))];table(s,5,12,['Task / candidate','Mean building WAPE'],pairs.map(x=>[x,null]));s.getRange('L:L').format.columnWidth=45;
pairs.forEach((pair,i)=>{const [task,model]=pair.split(' / ');s.getRange(`M${i+6}`).formulas=[[`=AVERAGEIFS($D$6:$D$29,$A$6:$A$29,"${task}",$C$6:$C$29,"${model}")`]];});s.getRange('M6:M13').setNumberFormat('0.00%');chart(s,'bar','L5:M13','Development selection (WAPE)','L16','W35',true);
note(s,'A3','Selection used only targets completed before 2017. Lower building-balanced WAPE wins.');
s=base('Example week',180,22);
table(s,5,1,['Local timestamp','Actual kW','Forecast kW','Last-week kW'],d.example.map(r=>[date(r.timestamp),r.actual_kw,r.predicted_kw,r.baseline_kw]));s.getRange('A:A').format.columnWidth=26;s.getRange('A6:A173').setNumberFormat('yyyy-mm-dd hh:mm');s.getRange('B6:D173').setNumberFormat('0.00');
table(s,5,6,['Local time','Actual kW','Forecast kW','Last-week kW'],d.example.map(r=>[r.timestamp.slice(5,10)+' '+r.timestamp.slice(11,16),null,null,null]));s.getRange('G6:I173').formulas=d.example.map((r,i)=>[r.actual_kw===null?'':`=B${i+6}`,r.predicted_kw===null?'':`=C${i+6}`,r.baseline_kw===null?'':`=D${i+6}`]);
const ec=chart(s,'line','F5:I173','Academic: first seven 2017 test days (kW)','K5','V27');ec.xAxis={axisType:'textAxis',tickLabelInterval:24};note(s,'A3','Predetermined example week, not a substitute for all-year results. Missing measurements stay blank.');s.freezePanes.freezeRows(5);
s=base('Data quality',d.monthly_quality.length+26,15);
table(s,5,1,['Building','Year','Grid hours','Eligible hours','Excluded hours'],d.quality.map(r=>[r.building,r.year,r.total_hours,r.eligible_hours,r.excluded_hours]));
table(s,5,7,['Building','Evaluation cutoff kW','Operational cutoff kW'],d.thresholds.map(r=>[r.building,r.evaluation_threshold_kw,r.operational_threshold_kw]));s.getRange('G:I').format.columnWidth=25;s.getRange('H6:I8').setNumberFormat('0.00');
table(s,24,1,['Building','Month','Measured mean kW','Valid hours','Expected hours','Coverage'],d.monthly_quality.map(r=>[r.building,r.month,r.mean_kw,r.valid_hours,r.expected_hours,r.coverage]));s.getRange(`C25:C${24+d.monthly_quality.length}`).setNumberFormat('0.00');s.getRange(`F25:F${24+d.monthly_quality.length}`).setNumberFormat('0.0%');
note(s,'G11','Source: user-supplied khusboo dataset .zip.');note(s,'G13','Each accepted hour has at least 48 valid minutes.');note(s,'G15','Each monthly target needs 80% hourly coverage.');note(s,'G17','Source timestamps use Asia/Kolkata local time.');note(s,'G19','No weather, tariff schedule or physical circuit labels.');
s=base('Overview',44,15);s.getRange('A:A').format.columnWidth=28;s.getRange('B:B').format.columnWidth=24;
table(s,5,1,['Forecast task','Selected model','MAE kW','WAPE','Baseline MAE kW'],[['Next-day hourly',d.hourly_metrics.selected,null,null,null],['Monthly leads 1–3',d.monthly_metrics.selected,null,null,null]]);
s.getRange('C6:E6').formulas=[[`=AVERAGE('Hourly predictions'!I6:I${hn})`,`=SUM('Hourly predictions'!I6:I${hn})/SUM('Hourly predictions'!C6:C${hn})`,`=AVERAGE('Hourly predictions'!J6:J${hn})`]];
s.getRange('C7:E7').formulas=[[`=AVERAGE('Monthly predictions'!H6:H${mn})`,`=SUM('Monthly predictions'!H6:H${mn})/SUM('Monthly predictions'!E6:E${mn})`,`=AVERAGE('Monthly predictions'!I6:I${mn})`]];
s.getRange('C6:C7').setNumberFormat('0.000');s.getRange('D6:D7').setNumberFormat('0.00%');s.getRange('E6:E7').setNumberFormat('0.000');
table(s,11,1,['Peak / non-peak metric','Result'],[['Accuracy',null],['Peak precision',null],['Peak recall',null],['Peak F1',null],['Macro F1',null]]);
table(s,11,8,['Actual / forecast','Non-peak','Peak'],[['Non-peak',null,null],['Peak',null,null]]);
for(let i=0;i<2;i++)for(let j=0;j<2;j++)s.getRange(`${String.fromCharCode(73+j)}${12+i}`).formulas=[[`=COUNTIFS('Hourly predictions'!$G$6:$G$${hn},${i},'Hourly predictions'!$H$6:$H$${hn},${j})`]];
s.getRange('B12:B16').formulas=[['=(I12+J13)/SUM(I12:J13)'],['=J13/SUM(J12:J13)'],['=J13/SUM(I13:J13)'],['=2*B13*B14/SUM(B13:B14)'],['=(B15+2*I12/(2*I12+I13+J12))/2']];s.getRange('B12:B14').setNumberFormat('0.00%');s.getRange('B15:B16').setNumberFormat('0.0000');
chart(s,'bar',['A5:A7','C5:C7','E5:E7'],'Forecast and baseline MAE (kW)','A20','G36');const pc=chart(s,'bar','A13:B15','Peak detection scores','H20','O36');pc.hasLegend=false;pc.yAxis={min:0,max:1,numberFormatCode:'0%',numberFormatSourceLinked:false};
note(s,'A3','Corrected Model 3: forecast demand and flag high-demand hours for three known buildings.');note(s,'A38','2017 forecasts use frozen pre-2017 models with historical inputs updated before each origin.');note(s,'A39','Monthly error is nearly tied with the seasonal baseline; the baseline has lower RMSE.');note(s,'A40','Future sheets show historical-origin January 2018 forecasts, not present-day forecasts.');note(s,'A41','Earlier building-recognition accuracy does not measure these forecasting tasks.');
wb.recalculate();
const checks=[['C6',d.hourly_metrics.overall.mae_kw],['D6',d.hourly_metrics.overall.wape],['C7',d.monthly_metrics.overall.mae_kw],['D7',d.monthly_metrics.overall.wape],['B15',d.hourly_metrics.peak.classification_report.peak['f1-score']]];
for(const [cell,value]of checks){const actual=s.getRange(cell).values[0][0];if(Math.abs(actual-value)>1e-9)throw Error('Excel metric mismatch '+cell+': '+actual);}
const input=sheets['Hourly predictions'].getRange('D6'),old=input.values[0][0],before=s.getRange('C6').values[0][0];input.values=[[old+100]];wb.recalculate();if(s.getRange('C6').values[0][0]===before)throw Error('Forecast edit did not update MAE');input.values=[[old]];wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Overview!A5:E7',include:'values,formulas',tableMaxRows:3,tableMaxCols:5,maxChars:1600})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1200})).ndjson);
await fs.mkdir(path.join(root,'previews'),{recursive:true});
const previewRanges={'Overview':'A1:O42','Future monthly':'A1:O30','Future hourly':'A1:P51','Hourly predictions':'A1:J13','Monthly predictions':'A1:I13','Monthly trends':'A1:O58','Seasonal errors':'A1:O26','Validation':'A1:W36','Example week':'A1:V28','Data quality':'A1:O31'};
const requestedPreviews=process.argv.slice(2);
for(const [name,range]of Object.entries(previewRanges)){if(requestedPreviews.length&&!requestedPreviews.includes(name))continue;const png=await wb.render({sheetName:name,range,scale:1,format:'png'});await fs.writeFile(path.join(root,'previews',name.replaceAll(' ','_')+'.png'),new Uint8Array(await png.arrayBuffer()));}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(root,'outputs/forecast_results.xlsx'));console.log('Saved forecast_results.xlsx');

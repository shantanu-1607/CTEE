// Run with the bundled Node runtime after summarize_results.py.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const root=path.dirname(fileURLToPath(import.meta.url));
const data=JSON.parse(await fs.readFile(path.join(root,'results/workbook_data.json'),'utf8'));
const out=path.join(root,'outputs/model3');
const previewDir=path.join(root,'workbook_previews');
await fs.mkdir(previewDir,{recursive:true});
const wb=Workbook.create();
const names=['Overview','Predictions','Validation','Comparison','Daily patterns','Test months','Features','Data quality'];
const sheets=Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const navy='#284B7D', green='#418474', orange='#CF7C35';
function write(s,row,col,rows){if(rows.length)s.getRangeByIndexes(row-1,col-1,rows.length,rows[0].length).values=rows;}
function letter(n){let s='';while(n){n--;s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26);}return s;}
function base(name,rows,cols){
 const s=sheets[name];s.showGridLines=false;
 const r=s.getRangeByIndexes(0,0,rows,cols);
 r.format.font={name:'Arial',size:10,color:'#182838'};r.format.rowHeight=23;r.format.columnWidth=17;r.format.verticalAlignment='center';
 s.getRange('A2').values=[[name]];s.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:navy};return s;
}
function table(s,row,headers,rows){write(s,row,1,[headers,...rows]);s.getRangeByIndexes(row-1,0,1,headers.length).format={fill:navy,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:38,wrapText:true,horizontalAlignment:'center',verticalAlignment:'center'};}
function note(s,cell,txt){s.getRange(cell).values=[[txt]];s.getRange(cell).format.font={name:'Arial',size:10,italic:true,color:'#465568'};}
function chart(s,type,ranges,title,start,end){
 const c=s.charts.add(type,Array.isArray(ranges)?ranges.map(r=>typeof r==='string'?s.getRange(r):r):s.getRange(ranges));
 c.title=title;c.titleTextStyle.fontSize=13;c.titleTextStyle.typeface='Arial';c.setPosition(start,end);
 c.legend={position:'top',textStyle:{typeface:'Arial',fontSize:10}};
 c.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:10}};
 c.yAxis={min:0,numberFormatCode:'0.00',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:10}};
 c.series.items.forEach((x,i)=>{x.fill=[navy,orange,green][i%3];if(type==='line')x.line={fill:[navy,orange,green][i%3],style:'solid',width:2};});
 if(type==='bar'){c.barOptions.direction='column';c.barOptions.grouping='clustered';}
 return c;
}

let s=base('Predictions',data.predictions.length+5,5);
const predictionEnd=data.predictions.length+5;
table(s,5,['Local date','Actual building','Predicted building','Correct (1/0)','Paired minutes'],data.predictions.map(r=>[new Date(r.date+'T00:00:00Z'),r.label,r.predicted,null,r.valid_minutes]));
s.getRange(`D6:D${predictionEnd}`).formulas=data.predictions.map((_,i)=>[`=IF(B${i+6}=C${i+6},1,0)`]);
s.getRange(`A6:A${predictionEnd}`).setNumberFormat('yyyy-mm-dd');
s.getRange('B:C').format.columnWidth=23;s.getRange(`D6:E${predictionEnd}`).setNumberFormat('0');s.freezePanes.freezeRows(5);
note(s,'A3','One prediction per retained building-day in the reserved later period.');

s=base('Validation',data.cv.length+5,10);
table(s,5,['Candidate','Fold','Training days','Validation days','Training ends','Validation starts','Validation ends','Accuracy','Macro F1','Seconds'],data.cv.map(r=>[r.model,r.fold,r.train_n,r.validation_n,new Date(r.train_end+'T00:00:00Z'),new Date(r.validation_start+'T00:00:00Z'),new Date(r.validation_end+'T00:00:00Z'),r.accuracy,r.macro_f1,r.fit_predict_seconds]));
s.getRange('A:A').format.columnWidth=32;s.getRange('E:G').format.columnWidth=21;s.getRange(`E6:G${data.cv.length+5}`).setNumberFormat('yyyy-mm-dd');s.getRange(`H6:H${data.cv.length+5}`).setNumberFormat('0.00%');s.getRange(`I6:J${data.cv.length+5}`).setNumberFormat('0.0000');s.freezePanes.freezeRows(5);
note(s,'A3','Three expanding time folds. Seven calendar days separate training and validation.');

s=base('Comparison',28,15);
table(s,5,['Candidate','Mean macro F1','Fold SD','Mean accuracy'],data.comparison.map(r=>[r.model,null,null,null]));
s.getRange('A:A').format.columnWidth=32;s.getRange('B:D').format.columnWidth=20;
for(let i=0;i<data.comparison.length;i++){
 const indices=data.cv.map((r,j)=>r.model===data.comparison[i].model?j+6:null).filter(x=>x!==null);
 const first=indices[0],last=indices.at(-1),row=i+6;
 s.getRange(`B${row}:D${row}`).formulas=[[`=AVERAGE(Validation!I${first}:I${last})`,`=STDEV(Validation!I${first}:I${last})`,`=AVERAGE(Validation!H${first}:H${last})`]];
}
s.getRange('B6:C10').setNumberFormat('0.0000');s.getRange('D6:D10').setNumberFormat('0.00%');
note(s,'A3','Ranked using mean development validation macro F1. Test scores did not select the model.');
const compChart=chart(s,'bar','A5:B10','Mean validation macro F1','F5','O24');compChart.yAxis={min:0,max:1};
note(s,'A26','Fold SD is available in column C; this editable chart displays the means.');

s=base('Daily patterns',29,15);
table(s,5,['Local hour','Academic (kW)','Facilities (kW)','Mess (kW)'],data.hourly.map(r=>[r.hour,r.Academic,r.Facilities,r.Mess]));
s.getRange('B6:D29').setNumberFormat('0.00');s.getRange('B:D').format.columnWidth=21;
note(s,'A3','Average of eligible daily hourly means in DEVELOPMENT data only. Asia/Kolkata.');
const patChart=chart(s,'line','A5:D29','Development daily load patterns (kW)','F5','O26');patChart.xAxis={axisType:'textAxis',tickLabelInterval:3};

s=base('Test months',28,15);
const monthEnd=data.monthly.length+5;
table(s,5,['Reserved month','Test days','Correct','Accuracy'],data.monthly.map(r=>[r.month,r.total,r.correct,null]));
s.getRange(`D6:D${monthEnd}`).formulas=data.monthly.map((_,i)=>[`=C${i+6}/B${i+6}`]);s.getRange(`D6:D${monthEnd}`).setNumberFormat('0.00%');
note(s,'A3','Test stability over time. All retained test days are represented.');
const monthChart=chart(s,'bar',[`A5:A${monthEnd}`,`D5:D${monthEnd}`],'Accuracy by reserved test month','F5','O24');monthChart.yAxis={min:0,max:1,numberFormatCode:'0%',numberFormatSourceLinked:false};

s=base('Features',data.importance.length+5,17);
table(s,5,['Electrical feature','Mean F1 drop','Shuffle SD','Unit','Definition'],data.importance.map(r=>[r.feature,r.mean_f1_drop,r.sd_f1_drop,r.unit,r.definition]));
s.getRange('A:A').format.columnWidth=34;s.getRange('B:C').format.columnWidth=19;s.getRange('D:D').format.columnWidth=12;s.getRange('E:E').format.columnWidth=68;
s.getRange(`E6:E${data.importance.length+5}`).format.wrapText=true;s.getRange(`A6:E${data.importance.length+5}`).format.rowHeight=44;s.getRange(`B6:C${data.importance.length+5}`).setNumberFormat('0.0000');s.freezePanes.freezeRows(5);
note(s,'A3','Importance: five shuffles on the last DEVELOPMENT validation fold. Correlated features share importance.');
const importanceChart=chart(s,'bar','A5:B17','Development validation feature importance','G5','Q24');importanceChart.yAxis={min:0,numberFormatCode:'0.000',numberFormatSourceLinked:false};

s=base('Data quality',28,8);
table(s,5,['Building','Development days','Embargo days','Test days'],data.counts.map(r=>[r.label,r.development,r.embargo,r.test]));
s.getRange('A:D').format.columnWidth=24;
table(s,12,['Building','Observed days','Eligible days','Rejected days'],data.quality.map(r=>[r.label,r.observed_days,r.eligible_days,null]));
s.getRange('D13:D15').formulas=data.quality.map((_,i)=>[`=B${i+13}-C${i+13}`]);
note(s,'A3','Counts are building-days; feature-identical duplicate exclusions occur after quality checks.');
note(s,'A18','Source: user-supplied khusboo dataset .zip. Original metadata: Readme.txt.');
note(s,'A20','Archive SHA-256: '+data.audit.archive_sha256);
note(s,'A22','Minimum coverage: 1,152 paired minutes and 20 hours with at least 30 paired minutes.');
note(s,'A24','Raw files stay outside Git. See results/dataset_audit.json for source-file details.');
note(s,'A26','These are three known buildings/meters. Unseen-building generalization was not measured.');

s=base('Overview',46,15);
s.getRange('A:A').format.columnWidth=31;s.getRange('B:D').format.columnWidth=21;
const intervals=data.metrics.approximate_week_block_bootstrap_95_ci;
table(s,5,['Reserved test metric','Result','95% lower','95% upper'],[
 ['Accuracy',null,intervals.accuracy.low,intervals.accuracy.high],
 ['Macro F1',null,intervals.macro_f1.low,intervals.macro_f1.high],
 ['Balanced accuracy',null,intervals.balanced_accuracy.low,intervals.balanced_accuracy.high],
 ['Test days',null,null,null],['Correct predictions',null,null,null],['Selected classifier',data.selection.selected_model,null,null],
 ['Validation macro F1',null,null,null],['Majority baseline accuracy',data.metrics.majority_baseline.accuracy,null,null]
]);
s.getRange('B6').formulas=[[`=SUM(Predictions!D6:D${predictionEnd})/COUNTA(Predictions!A6:A${predictionEnd})`]];
s.getRange('B7').formulas=[['=AVERAGE(D19:D21)']];s.getRange('B8').formulas=[['=AVERAGE(C19:C21)']];
s.getRange('B9').formulas=[[`=COUNTA(Predictions!A6:A${predictionEnd})`]];s.getRange('B10').formulas=[[`=SUM(Predictions!D6:D${predictionEnd})`]];
s.getRange('B12').formulas=[['=Comparison!B6']];
s.getRange('B6:D6').setNumberFormat('0.00%');s.getRange('B7:D7').setNumberFormat('0.0000');s.getRange('B8:D8').setNumberFormat('0.00%');s.getRange('B12').setNumberFormat('0.0000');s.getRange('B13').setNumberFormat('0.00%');
note(s,'A3','Recognize the source building from a completed daily electrical load profile.');
table(s,18,['Building','Precision','Recall','F1','Test days'],data.protocol.labels.map(x=>[x,null,null,null,null]));
s.getRange('B19:D21').setNumberFormat('0.0000');
write(s,18,8,[['Actual / predicted',...data.protocol.labels],...data.protocol.labels.map(x=>[x,null,null,null])]);
s.getRange('H18:K18').format={fill:navy,font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:38,wrapText:true};s.getRange('H:H').format.columnWidth=24;
for(let i=0;i<3;i++){
 const row=19+i;
 for(let j=0;j<3;j++)s.getRange(`${letter(9+j)}${row}`).formulas=[[`=COUNTIFS(Predictions!$B$6:$B$${predictionEnd},$H${row},Predictions!$C$6:$C$${predictionEnd},${letter(9+j)}$18)`]];
 const diag=`${letter(9+i)}${row}`;
 s.getRange(`B${row}:E${row}`).formulas=[[`=IF(SUM(${letter(9+i)}19:${letter(9+i)}21)=0,0,${diag}/SUM(${letter(9+i)}19:${letter(9+i)}21))`,`=${diag}/SUM(I${row}:K${row})`,`=IF(SUM(B${row}:C${row})=0,0,2*B${row}*C${row}/SUM(B${row}:C${row}))`,`=SUM(I${row}:K${row})`]];
}
const classChart=chart(s,'bar','A18:D21','Reserved test class scores','A25','G41');classChart.yAxis={min:0,max:1};
const countChart=chart(s,'bar',data.counts.length? [sheets['Data quality'].getRange('A5:A8'),sheets['Data quality'].getRange('B5:B8'),sheets['Data quality'].getRange('D5:D8')].map(x=>x):[],'Days by building','H25','O41');
note(s,'A43','Selected and saved before final test scoring. Test starts '+data.protocol.test_start+'.');
note(s,'A44','95% intervals: 1,000 weekly block bootstrap samples; approximate within this test period.');
note(s,'A45','One meter per class. Results apply to these buildings, not new buildings or physical R/I/RC/RLC types.');

wb.recalculate();
const overview=sheets.Overview;
for(const [cell,expected]of [['B6',data.metrics.accuracy],['B7',data.metrics.macro_f1],['B8',data.metrics.balanced_accuracy],['B9',data.metrics.n],['B10',data.metrics.correct]]){
 const got=overview.getRange(cell).values[0][0];if(Math.abs(got-expected)>1e-10)throw Error(`${cell}: expected ${expected}, got ${got}`);
}
// Verify a source edit changes derived metrics, then restore it.
const source=sheets.Predictions.getRange('C6');const old=source.values[0][0],actual=sheets.Predictions.getRange('B6').values[0][0];
source.values=[[old===actual?data.protocol.labels.find(x=>x!==actual):actual]];wb.recalculate();
if(overview.getRange('B6').values[0][0]===data.metrics.accuracy)throw Error('Prediction edit did not change accuracy');
source.values=[[old]];wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Overview!A5:D13',include:'values,formulas',tableMaxRows:9,tableMaxCols:4,maxChars:2200})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1200})).ndjson);
const previews={Overview:'A1:O46',Predictions:'A1:E13',Validation:'A1:J12',Comparison:'A1:O27','Daily patterns':'A1:O29','Test months':'A1:O25',Features:'A1:Q25','Data quality':'A1:H27'};
for(const [name,range]of Object.entries(previews)){
 const png=await wb.render({sheetName:name,range,scale:1,format:'png'});
 await fs.writeFile(path.join(previewDir,name.replaceAll(' ','_')+'.png'),new Uint8Array(await png.arrayBuffer()));
}
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'chart_data.xlsx'));
console.log('Exported Model 3 workbook');

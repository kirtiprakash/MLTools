/* MLTools · Data Splitting and Validation Lab
   All examples are deterministic synthetic teaching data, generated locally.
   No remote libraries, cookies, analytics or external datasets. */
'use strict';
const $=id=>document.getElementById(id);
const COLORS={train:'#168a97',validation:'#e7a23b',test:'#8b75c9',majority:'#187d9b',minority:'#e36e55'};
const defaults={n:150,minority:20,train:60,validation:20,method:'random',seed:42,folds:5,foldType:'stratified',fold:0,modelK:3};
let s={...defaults}, data=[],groups={},split={},cv=null,unlocked=false;
const fmt=(v,d=0)=>Number(v).toFixed(d);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function rng(seed){let a=seed>>>0;return ()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
function shuffle(list,random){const a=list.slice();for(let i=a.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function normal(random){const a=Math.max(1e-9,random()),b=random();return Math.sqrt(-2*Math.log(a))*Math.cos(2*Math.PI*b);}
function synthetic(){
 const r=rng(2026+s.n*31+s.minority*97),nRare=Math.round(s.n*s.minority/100),a=[];
 for(let i=0;i<s.n;i++){
  const label=i<nRare?1:0;
  const x=clamp((label?6.0:4.1)+1.55*normal(r),.3,9.7);
  const y=clamp((label?5.6:4.4)+1.55*normal(r),.3,9.7);
  a.push({id:i+1,x,y,label});
 }
 return shuffle(a,r);
}
function apportion(total,shares){
 const exact=shares.map(v=>v*total),counts=exact.map(Math.floor),remaining=total-counts.reduce((a,b)=>a+b,0);
 const order=exact.map((v,i)=>({i,frac:v-counts[i]})).sort((a,b)=>b.frac-a.frac||a.i-b.i);
 for(let j=0;j<remaining;j++)counts[order[j].i]++;
 return counts;
}
function splitData(){
 const shares=[s.train,s.validation,100-s.train-s.validation].map(x=>x/100);
 const capacities=apportion(data.length,shares),out=[[],[],[]];
 const r=rng(s.seed);
 if(s.method==='random'){
  const shuffled=shuffle(data,r);let start=0;
  for(let i=0;i<3;i++){out[i]=shuffled.slice(start,start+capacities[i]);start+=capacities[i];}
 }else{
  const groups=[shuffle(data.filter(p=>p.label===0),r),shuffle(data.filter(p=>p.label===1),r)];
  const rare=groups[1].length;
  const minorityBySplit=apportion(rare,capacities.map(v=>v/data.length));
  // Global capacities are exact; the majority class fills remaining spaces.
  for(let i=0,j=0;i<3;i++){
   out[i].push(...groups[1].slice(j,j+minorityBySplit[i]));j+=minorityBySplit[i];
  }
  for(let i=0,j=0;i<3;i++){
   const target=capacities[i]-out[i].length;
   out[i].push(...groups[0].slice(j,j+target));j+=target;
   out[i]=shuffle(out[i],rng(s.seed+89*i+411));
  }
 }
 split={train:out[0],validation:out[1],test:out[2]};
 // Expose assignments to export exactly the plot the students see.
 groups=Object.fromEntries(Object.entries(split).flatMap(([set,pts])=>pts.map(p=>[p.id,set])));
}
function counts(pts){return {total:pts.length,majority:pts.filter(p=>p.label===0).length,minority:pts.filter(p=>p.label===1).length};}
function context(canvas){
 const r=canvas.getBoundingClientRect(),d=Math.min(2,window.devicePixelRatio||1);
 canvas.width=Math.round(Math.max(r.width,10)*d);canvas.height=Math.round(Math.max(r.height,10)*d);
 const cx=canvas.getContext('2d');cx.setTransform(d,0,0,d,0,0);return [cx,r.width,r.height];
}
function scatter(canvas,pts){
 const [ctx,w,h]=context(canvas);ctx.fillStyle='#fbfdfe';ctx.fillRect(0,0,w,h);
 const L=27,R=12,T=11,B=29;
 const x=v=>L+v/10*(w-L-R),y=v=>h-B-v/10*(h-T-B);
 for(let tick=0;tick<=10;tick+=2){
  ctx.strokeStyle='#e2eaee';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(x(tick),T);ctx.lineTo(x(tick),h-B);ctx.moveTo(L,y(tick));ctx.lineTo(w-R,y(tick));ctx.stroke();
  ctx.fillStyle='#718693';ctx.font='10px system-ui';ctx.textAlign='center';ctx.fillText(tick,x(tick),h-11);
 }
 ctx.strokeStyle='#8ba0ad';ctx.lineWidth=1.2;ctx.strokeRect(L,T,w-L-R,h-T-B);
 for(const p of pts){ctx.fillStyle=p.label?COLORS.minority:COLORS.majority;ctx.beginPath();ctx.arc(x(p.x),y(p.y),3.2,0,Math.PI*2);ctx.fill();}
}
function progressBar(id,pct,color){const e=$(id);e.style.width=clamp(pct,0,100)+'%';e.style.background=color;}
function renderSplit(){
 $('split-label').textContent=s.train+'% train · '+s.validation+'% validation · '+(100-s.train-s.validation)+'% test';
 $('flow-train').style.width=s.train+'%';
 $('flow-validation').style.width=s.validation+'%';
 $('flow-test').style.width=(100-s.train-s.validation)+'%';
 for(const [key,name] of [['train','Training'],['validation','Validation'],['test','Test']]){
  const list=split[key],c=counts(list),pct=c.total?Math.round(c.minority/c.total*100):0;
  $('num-'+key).textContent=c.total;
  $('rare-'+key).textContent=pct+'%';
  $('count-'+key).textContent=c.majority+' common · '+c.minority+' rare';
  scatter($('scatter-'+key),list);
  progressBar('bar-'+key,pct,COLORS.minority);
 }
 const rare=counts(data).minority/data.length*100;
 $('target-minority').textContent=fmt(rare,1)+'%';
 $('split-explainer').textContent=s.method==='stratified'?
  'Stratified splitting preserves the overall class proportions as closely as possible within each set (subject to rounding).':
  'Random splitting preserves the overall sizes but does not enforce class balance. Try raising the class imbalance and changing the random seed.';
 $('total-label').textContent=data.length;
 $('sums-check').textContent=Object.values(split).reduce((sum,part)=>sum+part.length,0)+' / '+data.length+' examples allocated';
 $('split-seed-label').textContent='Seed: '+s.seed;
 const a=counts(split.train),b=counts(split.validation),c=counts(split.test);
 $('split-difference').textContent='Rare-class proportions: '+fmt(a.minority/a.total*100,1)+'% / '+fmt(b.minority/b.total*100,1)+'% / '+fmt(c.minority/c.total*100,1)+'% (train / validation / test).';
 $('rare-warning').textContent=[a,b,c].some(group=>group.minority===0)
  ? 'One or more subsets contain no rare-class examples. With very few rare observations, even stratification cannot guarantee a representative sample in every subset. Increase the dataset size or reconsider the evaluation design.'
  : '';
}
function nearest(train,point,k){
 return train.map(p=>({p,d:(p.x-point.x)**2+(p.y-point.y)**2}))
 .sort((a,b)=>a.d-b.d||a.p.id-b.p.id).slice(0,Math.min(k,train.length));
}
function classify(train,point,k){
 const ns=nearest(train,point,k);if(!ns.length)return null;
 const votes=[0,0];ns.forEach(o=>votes[o.p.label]++);
 return votes[1]===votes[0]?null:(votes[1]>votes[0]?1:0);
}
function accuracy(train,valid,k){
 if(!valid.length||!train.length)return NaN;
 let correct=0;
 for(const p of valid)if(classify(train,p,k)===p.label)correct++;
 return correct/valid.length;
}
function foldBuckets(train){
 const r=rng(s.seed+713),bucket=Array.from({length:s.folds},()=>[]);
 if(s.foldType==='stratified'){
  let offset=0;
  for(const label of [0,1]){
   const shuffled=shuffle(train.filter(p=>p.label===label),r);
   shuffled.forEach((p,i)=>bucket[(i+offset)%s.folds].push(p));
   offset=(offset+shuffled.length)%s.folds;
  }
 }else{
  shuffle(train,r).forEach((p,i)=>bucket[i%s.folds].push(p));
 }
 return bucket;
}
function buildCV(){
 const bucket=foldBuckets(split.train);
 const scores=bucket.map((held,j)=>{
  const train=bucket.flatMap((part,i)=>i===j?[]:part);
  return accuracy(train,held,s.modelK);
 });
 cv={bucket,scores,mean:scores.reduce((x,y)=>x+y,0)/scores.length};
}
function renderCV(){
 buildCV();s.fold=clamp(s.fold,0,s.folds-1);
 $('cv-method').textContent=s.foldType==='stratified'?'Stratified K-Fold':'Ordinary K-Fold';
 $('cv-fold-number').textContent=(s.fold+1)+' / '+s.folds;
 $('cv-grid').replaceChildren();
 const fragment=document.createDocumentFragment();
 for(let j=0;j<s.folds;j++){
  const item=document.createElement('button');
  item.type='button';item.className='fold-row'+(s.fold===j?' chosen':'');item.setAttribute('aria-label','View fold '+(j+1));
  const n=document.createElement('span');n.className='fold-id';n.textContent='Fold '+(j+1);
  item.appendChild(n);
  const blocks=document.createElement('span');blocks.className='fold-blocks';
  for(let col=0;col<s.folds;col++){
   const seg=document.createElement('i');seg.className='fold-segment '+(col===j?'held':'learn');
   seg.title=col===j?'Validation subset':'Training subset';blocks.appendChild(seg);
  }
  item.appendChild(blocks);
  const score=document.createElement('span');score.className='fold-score';score.textContent=fmt(cv.scores[j]*100,1)+'%';
  item.appendChild(score);
  item.addEventListener('click',()=>{s.fold=j;renderCVHighlight();});
  fragment.appendChild(item);
 }
 $('cv-grid').appendChild(fragment);
 renderCVHighlight();
 const scores=cv.scores;
 $('cv-mean').textContent=fmt(cv.mean*100,1)+'%';
 $('cv-range').textContent=fmt(Math.min(...scores)*100,1)+'–'+fmt(Math.max(...scores)*100,1)+'%';
 $('val-score').textContent=fmt(accuracy(split.train,split.validation,s.modelK)*100,1)+'%';
 $('modelk-label').textContent=s.modelK;
 $('fold-count-label').textContent=s.folds;
 $('cv-equation').textContent='Mean validation accuracy = ('+scores.map(v=>fmt(v*100,1)).join(' + ')+') / '+s.folds+' = '+fmt(cv.mean*100,1)+'%';
}
function renderCVHighlight(){
 [...$('cv-grid').children].forEach((e,i)=>e.classList.toggle('chosen',i===s.fold));
 $('cv-fold-number').textContent=(s.fold+1)+' / '+s.folds;
 const held=cv.bucket[s.fold],fitted=cv.bucket.flatMap((p,i)=>i===s.fold?[]:p);
 $('fold-interpretation').textContent='Fold '+(s.fold+1)+' holds out '+held.length+' training-pool observations, fits on the remaining '+fitted.length+', and scores '+fmt(cv.scores[s.fold]*100,1)+'% on held-out observations.';
}
function redrawTests(){
 $('test-result').textContent='Locked until you deliberately reveal it.';
 $('test-result').classList.remove('unlocked');
 $('test-reveal').disabled=false;unlocked=false;
}
function revealTest(){
 const refit=split.train.concat(split.validation);
 const result=accuracy(refit,split.test,s.modelK);
 $('test-result').textContent='Final held-out test accuracy: '+fmt(result*100,1)+'% ('+Math.round(result*split.test.length)+'/'+split.test.length+' correct). The model was refitted using the training and validation sets; test data were not used to choose K.';
 $('test-result').classList.add('unlocked');$('test-reveal').disabled=true;unlocked=true;
}
function makeLeakageExample(){
 const r=rng(27015),subjects=[];
 for(let i=0;i<80;i++){
  const x=.5+9*r(),y=.5+9*r(),label=r()<.5?0:1;
  for(let j=0;j<4;j++)subjects.push({id:4*i+j+1,group:i+1,x:x+normal(r)*.004,y:y+normal(r)*.004,label});
 }
 const rows=shuffle(subjects,rng(13884)),nTrain=Math.floor(rows.length*.8);
 const naiveTrain=rows.slice(0,nTrain),naiveTest=rows.slice(nTrain);
 const ids=shuffle(Array.from({length:80},(_,i)=>i+1),rng(13884));
 const trainingGroups=new Set(ids.slice(0,64));
 const groupedTrain=subjects.filter(p=>trainingGroups.has(p.group));
 const groupedTest=subjects.filter(p=>!trainingGroups.has(p.group));
 const overlap=naiveTest.filter(p=>naiveTrain.some(t=>t.group===p.group)).length;
 return {
  rowAccuracy:accuracy(naiveTrain,naiveTest,1),
  groupAccuracy:accuracy(groupedTrain,groupedTest,1),
  overlap,naiveTest:naiveTest.length,groupedTest:groupedTest.length
 };
}
function renderLeakage(){
 const l=makeLeakageExample();
 $('leaky-accuracy').textContent=fmt(l.rowAccuracy*100,1)+'%';
 $('honest-accuracy').textContent=fmt(l.groupAccuracy*100,1)+'%';
 progressBar('leaky-bar',l.rowAccuracy*100,COLORS.minority);
 progressBar('honest-bar',l.groupAccuracy*100,COLORS.train);
 $('duplicate-count').textContent=l.overlap+' of '+l.naiveTest;
 $('leakage-note').textContent='In the random row split, '+l.overlap+' of '+l.naiveTest+' test rows share a source subject with training. In the group-aware split, zero do. The features are near-identical repeats and the subject labels are independent of location. Apparent accuracy is therefore inflated for the wrong evaluation question.';
}
function buildCode(){
 const strat=s.method==='stratified',test=100-s.train-s.validation;
 const snippet=[
'# Export splitting_lab.csv from the interactive experiment first.',
'import pandas as pd',
'from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score',
'from sklearn.pipeline import make_pipeline',
'from sklearn.preprocessing import StandardScaler',
'from sklearn.neighbors import KNeighborsClassifier',
'from sklearn.metrics import accuracy_score',
'',
"df = pd.read_csv('splitting_lab.csv')",
"X = df[['feature_1', 'feature_2']]",
"y = df['label']  # 1 = rarer class",
'',
'# Hold out the test set BEFORE tuning any hyperparameters.',
'X_build, X_test, y_build, y_test = train_test_split(',
'    X, y, test_size='+fmt(test/100,3)+', random_state='+s.seed+',',
'    stratify='+(strat?'y':'None')+')',
'# Split the remaining samples into training and validation sets.',
'X_train, X_val, y_train, y_val = train_test_split(',
'    X_build, y_build, test_size='+fmt(s.validation/(s.train+s.validation),4)+',',
'    random_state='+s.seed+',',
'    stratify='+(strat?'y_build':'None')+')',
'',
'# Keep scaling inside a Pipeline to avoid leakage between folds.',
'pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors='+s.modelK+'))',
'cv = '+(s.foldType==='stratified'?'StratifiedKFold':'KFold')+'(',
'    n_splits='+s.folds+', shuffle=True, random_state='+s.seed+')',
'scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")',
'print("CV scores:", scores.round(3))',
'print("Mean CV accuracy:", scores.mean().round(3))',
'',
'# Validation set can be used for model-selection decisions.',
'pipeline.fit(X_train, y_train)',
'print("Validation accuracy:", accuracy_score(y_val, pipeline.predict(X_val)).round(3))',
'',
'# After finalising K: refit on training + validation, evaluate test ONCE.',
'from pandas import concat',
'X_final = concat([X_train, X_val]); y_final = concat([y_train, y_val])',
'pipeline.fit(X_final, y_final)',
'print("Final test accuracy:", accuracy_score(y_test, pipeline.predict(X_test)).round(3))'
 ];
 $('python-code').textContent=snippet.join('\n');
 $('code-note').textContent='The app and Python both demonstrate '+s.method+' splitting. Python uses scikit-learn’s own random generator, so individual rows may be assigned differently from the browser even with the same seed; proportions and leakage safeguards remain comparable.';
}
function saveCSV(){
 const header='id,feature_1,feature_2,label,split';
 const lines=data.map(p=>[p.id,fmt(p.x,6),fmt(p.y,6),p.label,groups[p.id]].join(','));
 const url=URL.createObjectURL(new Blob([[header,...lines].join('\n')+'\n'],{type:'text/csv;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download='splitting_lab.csv';document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),1300);
}
function copyCode(){
 const node=$('copy-code');
 navigator.clipboard.writeText($('python-code').textContent)
 .then(()=>{node.textContent='Copied!';setTimeout(()=>node.textContent='Copy Python code',1800);})
 .catch(()=>{node.textContent='Select the code below to copy';});
}
function refresh(newData=false){
 if(newData)data=synthetic();
 splitData();renderSplit();renderCV();redrawTests();buildCode();
 const test=100-s.train-s.validation;
 $('train-slider').value=s.train;$('validation-slider').value=s.validation;
 $('validation-slider').max=90-s.train;
 $('train-output').textContent=s.train+'%';$('validation-output').textContent=s.validation+'%';
 $('test-output').textContent=test+'%';
 $('n-output').textContent=s.n;$('minority-output').textContent=s.minority+'%';
 $('seed-input').value=s.seed;$('split-method').value=s.method;
 $('folds-slider').value=s.folds;$('folds-output').textContent=s.folds;
 $('foldtype').value=s.foldType;$('model-k').value=s.modelK;
}
$('n-slider').oninput=e=>{s.n=Number(e.target.value);refresh(true);};
$('minority-slider').oninput=e=>{s.minority=Number(e.target.value);refresh(true);};
$('train-slider').oninput=e=>{s.train=Number(e.target.value);s.validation=Math.min(s.validation,90-s.train);refresh();};
$('validation-slider').oninput=e=>{s.validation=Number(e.target.value);refresh();};
$('split-method').onchange=e=>{s.method=e.target.value;refresh();};
$('seed-input').onchange=e=>{s.seed=clamp(Math.round(Number(e.target.value)||0),0,999999);refresh();};
$('reshuffle').onclick=()=>{s.seed=(s.seed+1)%1000000;refresh();};
$('reset-all').onclick=()=>{s={...defaults};refresh(true);};
$('folds-slider').oninput=e=>{s.folds=Number(e.target.value);s.fold=0;renderCV();redrawTests();buildCode();$('folds-output').textContent=s.folds;};
$('foldtype').onchange=e=>{s.foldType=e.target.value;renderCV();redrawTests();buildCode();};
$('model-k').onchange=e=>{s.modelK=Number(e.target.value);renderCV();redrawTests();buildCode();};
$('next-fold').onclick=()=>{s.fold=(s.fold+1)%s.folds;renderCVHighlight();};
$('previous-fold').onclick=()=>{s.fold=(s.fold+s.folds-1)%s.folds;renderCVHighlight();};
$('test-reveal').onclick=revealTest;
$('export-csv').onclick=saveCSV;
$('copy-code').onclick=copyCode;
let pending=null;
window.addEventListener('resize',()=>{if(pending)cancelAnimationFrame(pending);pending=requestAnimationFrame(()=>{renderSplit();});});
renderLeakage();refresh(true);

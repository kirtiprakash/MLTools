/* MLTools · K-nearest neighbours teaching laboratory
   All data generated locally. No tracking, libraries or network calls. */
'use strict';
const $ = id => document.getElementById(id);
const PALETTE = ['#167b9a','#e26b54'];
const CONFIG = {
 flowers:{title:'Flower garden',xlabel:'Sepal length (cm)',ylabel:'Petal width (cm)',range:[4,8,0,2.6],classes:['Setosa-like','Other iris'],query:[5.75,1.06],seed:1244},
 patients:{title:'Illustrative patient records',xlabel:'Malignant nodes',ylabel:'Age (years)',range:[0,30,20,80],classes:['Survived','Did not survive'],query:[11,42],seed:2831},
 study:{title:'Study habits',xlabel:'Study hours / week',ylabel:'Practice questions',range:[0,12,0,90],classes:['Needs more practice','Likely pass'],query:[5.6,42],seed:7007}
};
let type='flowers', points=[], query=[], serial=0, version=0, reveal=0, dragging=false;
let boundaryCache=null, looCache=null, modelSnapshot=null;
const state={k:5,metric:'euclidean',scaling:'standard',weights:'uniform',boundary:true,rings:true,mode:'query',labelNoise:8};
const rnd=seed=>()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
const fmt=(n,d=2)=>Number(n).toFixed(d);
function dataset(which){
 const c=CONFIG[which],r=rnd(c.seed),p=[];
 if(which==='flowers'){
  for(let i=0;i<50;i++){
   const label=i<25?0:1;
   // Deliberately overlapping synthetic classes to reveal underfitting and overfitting.
   const x=(label===0?5.15:6.15)+(r()+r()-1)*1.4;
   const y=(label===0?0.84:1.42)+(r()+r()-1)*1.1;
   p.push({x:clamp(x,4.1,7.9),y:clamp(y,0.06,2.5),label,id:i});
  }
 } else if(which==='patients'){
  for(let i=0;i<52;i++){
   const label=i<28?0:1;
   const x=label===0?1+r()*16:8+r()*20;
   const y=label===0?23+r()*40:32+r()*44;
   p.push({x:clamp(x,0.4,29.3),y:clamp(y,21,78),label,id:i});
  }
 }else{
  for(let i=0;i<55;i++){
   const x=0.8+r()*10.5,y=8+r()*76,noise=(r()-.5)*27;
   const label=5.2*x+0.58*y+noise>51?1:0;
   p.push({x,y,label,id:i});
  }
 }
 // Inject a controlled amount of label noise. Positions stay identical across settings.
 // A separate seeded shuffle means changing noise affects labels, not the geometry.
 const noiseRnd=rnd(c.seed+921),indices=Array.from({length:p.length},(_,i)=>i);
 for(let i=indices.length-1;i>0;i--){
  const j=Math.floor(noiseRnd()*(i+1));
  [indices[i],indices[j]]=[indices[j],indices[i]];
 }
 const corrupted=Math.round(p.length*state.labelNoise/100);
 for(let i=0;i<corrupted;i++)p[indices[i]].label=1-p[indices[i]].label;
 return p;
}
function featureStats(train){
 const n=train.length||1;
 const mx=train.reduce((a,p)=>a+p.x,0)/n,my=train.reduce((a,p)=>a+p.y,0)/n;
 const sx=Math.sqrt(train.reduce((a,p)=>a+(p.x-mx)**2,0)/n)||1;
 const sy=Math.sqrt(train.reduce((a,p)=>a+(p.y-my)**2,0)/n)||1;
 return {sx,sy,mx,my};
}
function distance(a,b,stats){
 const dx=(a.x-b.x)/(state.scaling==='standard'?stats.sx:1);
 const dy=(a.y-b.y)/(state.scaling==='standard'?stats.sy:1);
 return state.metric==='manhattan'?Math.abs(dx)+Math.abs(dy):Math.hypot(dx,dy);
}
function closest(test,train,k,stats){
 return train.map(p=>({p,d:distance(test,p,stats)}))
  .sort((a,b)=>a.d-b.d||a.p.id-b.p.id).slice(0,Math.min(k,train.length));
}
function vote(neighbors){
 const score=[0,0],count=[0,0];
 const exact=state.weights==='distance'&&neighbors.some(o=>o.d<1e-10);
 for(const o of neighbors){
  count[o.p.label]++;
  const w=state.weights==='uniform'?1:(exact?(o.d<1e-10?1:0):1/Math.max(o.d,1e-10));
  score[o.p.label]+=w;
 }
 const same=Math.abs(score[0]-score[1])<1e-8*Math.max(1,score[0],score[1]);
 return {count,score,pred:same?null:score[0]>score[1]?0:1};
}
function prediction(test=query,k=state.k,train=points,stats=featureStats(train)){
 const neighbors=closest({x:test[0],y:test[1]},train,k,stats);
 return {...vote(neighbors),neighbors};
}
function resizeCanvas(canvas){
 const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;
 const d=Math.min(2,window.devicePixelRatio||1);
 const pw=Math.max(1,Math.round(w*d)),ph=Math.max(1,Math.round(h*d));
 if(canvas.width!==pw||canvas.height!==ph){canvas.width=pw;canvas.height=ph;boundaryCache=null;}
 const ctx=canvas.getContext('2d');ctx.setTransform(d,0,0,d,0,0);
 return {ctx,w,h};
}
function geometry(w,h){
 const l=61,r=18,t=18,b=50,span=CONFIG[type].range;
 return {l,r,t,b,xmin:span[0],xmax:span[1],ymin:span[2],ymax:span[3],
  X:x=>l+(x-span[0])/(span[1]-span[0])*(w-l-r),
  Y:y=>h-b-(y-span[2])/(span[3]-span[2])*(h-t-b),
  unX:px=>span[0]+(px-l)/(w-l-r)*(span[1]-span[0]),
  unY:py=>span[2]+(h-b-py)/(h-t-b)*(span[3]-span[2])
 };
}
function makeBoundary(w,h,g){
 const key=[version,type,state.k,state.metric,state.scaling,state.weights,w,h].join('|');
 if(boundaryCache&&boundaryCache.key===key)return boundaryCache.canvas;
 const c=document.createElement('canvas');c.width=Math.ceil(w);c.height=Math.ceil(h);
 const cx=c.getContext('2d');const s=featureStats(points);const step=12;
 for(let x=g.l;x<w-g.r;x+=step)for(let y=g.t;y<h-g.b;y+=step){
  const t=[g.unX(x+step/2),g.unY(y+step/2)],pred=prediction(t,state.k,points,s).pred;
  cx.fillStyle=pred===null?'#d6d9db':pred===0?'#dcedf2':'#fbe5de';
  cx.fillRect(x,y,step+1,step+1);
 }
 boundaryCache={key,canvas:c};return c;
}
function render(){
 const canvas=$('plot'),{ctx,w,h}=resizeCanvas(canvas),g=geometry(w,h),c=CONFIG[type];
 ctx.clearRect(0,0,w,h);ctx.fillStyle='#fbfcfd';ctx.fillRect(0,0,w,h);
 if(state.boundary)ctx.drawImage(makeBoundary(w,h,g),0,0,w,h);
 ctx.font='12px system-ui,sans-serif';ctx.strokeStyle='#d6e0e5';ctx.lineWidth=1;
 for(let i=0;i<=5;i++){
  const x=g.xmin+(g.xmax-g.xmin)*i/5,y=g.ymin+(g.ymax-g.ymin)*i/5;
  ctx.beginPath();ctx.moveTo(g.X(x),g.t);ctx.lineTo(g.X(x),h-g.b);
  ctx.moveTo(g.l,g.Y(y));ctx.lineTo(w-g.r,g.Y(y));ctx.stroke();
  ctx.fillStyle='#546776';ctx.textAlign='center';ctx.fillText(fmt(x,1),g.X(x),h-g.b+19);
  ctx.textAlign='right';ctx.fillText(fmt(y,1),g.l-10,g.Y(y)+4);
 }
 ctx.strokeStyle='#7d919e';ctx.lineWidth=1.4;ctx.strokeRect(g.l,g.t,w-g.l-g.r,h-g.t-g.b);
 ctx.textAlign='center';ctx.fillStyle='#243b4e';ctx.font='12px system-ui,sans-serif';
 ctx.fillText(c.xlabel,(w+g.l-g.r)/2,h-9);
 ctx.save();ctx.translate(16,h/2);ctx.rotate(-Math.PI/2);ctx.fillText(c.ylabel,0,0);ctx.restore();
 const stats=featureStats(points),full=prediction(query,state.k,points,stats),active=reveal>0?full.neighbors.slice(0,reveal):full.neighbors;
 if(state.rings){
  ctx.lineWidth=1.7;ctx.setLineDash([5,4]);ctx.strokeStyle='#718f8d';
  for(const item of active){ctx.beginPath();ctx.moveTo(g.X(query[0]),g.Y(query[1]));ctx.lineTo(g.X(item.p.x),g.Y(item.p.y));ctx.stroke();}
  ctx.setLineDash([]);
 }
 for(const p of points){
  ctx.beginPath();ctx.arc(g.X(p.x),g.Y(p.y),5.4,0,Math.PI*2);
  ctx.fillStyle=PALETTE[p.label];ctx.fill();
  ctx.strokeStyle='#ffffff';ctx.lineWidth=1.2;ctx.stroke();
 }
 if(state.rings)for(let i=0;i<active.length;i++){
  const p=active[i].p;
  ctx.beginPath();ctx.arc(g.X(p.x),g.Y(p.y),10,0,Math.PI*2);
  ctx.strokeStyle='#267a52';ctx.lineWidth=2.4;ctx.stroke();
 }
 const px=g.X(query[0]),py=g.Y(query[1]);ctx.beginPath();ctx.arc(px,py,11,0,Math.PI*2);
 ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle='#172a40';ctx.lineWidth=3;ctx.stroke();
 ctx.beginPath();ctx.arc(px,py,3.5,0,Math.PI*2);ctx.fillStyle='#172a40';ctx.fill();
 $('plot-caption').textContent=state.mode==='query'?'Click or drag on the chart to move the unknown point.':'Click the chart to add a new '+c.classes[state.mode==='add0'?0:1]+' observation.';
 updateReadout(full,active,stats);
}
function updateReadout(full,active,stats){
 const c=CONFIG[type],v=vote(active),isPartial=reveal>0&&reveal<state.k;
 $('prediction-name').textContent=v.pred===null?'Tie — try another K':c.classes[v.pred];
 $('prediction-note').textContent=isPartial?'Partial vote: '+active.length+' of '+state.k+' neighbours revealed.':v.pred===null?'An exact vote tie has no unique majority.':'Based on '+active.length+' closest labelled examples';
 $('prediction-swatch').style.background=v.pred===null?'#89959e':PALETTE[v.pred];
 $('votes-a-label').textContent=c.classes[0];$('votes-b-label').textContent=c.classes[1];
 $('votes-a-num').textContent=state.weights==='uniform'?v.count[0]:fmt(v.score[0]);
 $('votes-b-num').textContent=state.weights==='uniform'?v.count[1]:fmt(v.score[1]);
 const total=v.score[0]+v.score[1]||1;
 $('votes-a-bar').style.width=fmt(100*v.score[0]/total,1)+'%';
 $('votes-b-bar').style.width=fmt(100*v.score[1]/total,1)+'%';
 $('k-output').textContent=state.k;
 $('point-x').value=fmt(query[0],2);$('point-y').value=fmt(query[1],2);
 $('shown-count').textContent=reveal>0?'Revealed '+reveal+' of '+state.k:'All '+state.k+' selected';
 $('next-neighbour').textContent=reveal>=state.k?'Start vote again':'Reveal next neighbour';
 $('nearest-rows').replaceChildren(...full.neighbors.slice(0,Math.min(8,state.k)).map((item,i)=>{
  const tr=document.createElement('tr');const cells=[i+1,c.classes[item.p.label],fmt(item.p.x),fmt(item.p.y),fmt(item.d,3)];
  for(const val of cells){const td=document.createElement('td');td.textContent=val;tr.appendChild(td);}
  return tr;
 }));
 $('nearest-summary').textContent=state.k>8?'Showing the first 8 of '+state.k+' neighbours.':'Sorted from nearest to farthest.';
 const first=full.neighbors[0];
 if(first){
  let dx=Math.abs(query[0]-first.p.x),dy=Math.abs(query[1]-first.p.y);
  const sx=state.scaling==='standard'?stats.sx:1,sy=state.scaling==='standard'?stats.sy:1;
  const ax=dx/sx,ay=dy/sy,metric=state.metric==='euclidean'?'√('+fmt(ax,2)+'² + '+fmt(ay,2)+'²)':'|'+fmt(ax,2)+'| + |'+fmt(ay,2)+'|';
  $('worked-example').textContent='Nearest example: Δ'+c.xlabel.split(' (')[0]+' = '+fmt(dx,2)+', Δ'+c.ylabel.split(' (')[0]+' = '+fmt(dy,2)+'.  d = '+metric+' = '+fmt(first.d,3)+'.';
 }
 $('distance-note').textContent=state.scaling==='standard'?'Distances use (difference / training standard deviation) for each feature. The plotted axes remain in original units.':'Distances use original feature units. A feature with large numeric values may dominate the result.';
 $('metric-equation').textContent=state.metric==='euclidean'?'d(a,b) = √[(a₁ − b₁)² + (a₂ − b₂)²]':'d(a,b) = |a₁ − b₁| + |a₂ − b₂|';
 $('weight-note').textContent=state.weights==='distance'?'Closer neighbours receive more influence: weight ≈ 1 / distance.':'Each of the K neighbours casts one equal vote.';
}
function looScores(){
 const key=[version,type,state.metric,state.scaling,state.weights].join('|');
 if(looCache&&looCache.key===key)return looCache;
 const maximum=Math.min(25,points.length-1);
 const validHits=Array(maximum).fill(0),trainHits=Array(maximum).fill(0);
 const trainingStats=featureStats(points);
 for(let i=0;i<points.length;i++){
  const p=points[i],other=points.filter((_,j)=>i!==j);
  // No leakage: scale every held-out point using statistics from the other rows.
  const neighbors=closest(p,other,maximum,featureStats(other));
  const trainNeighbors=closest(p,points,maximum,trainingStats);
  for(let k=1;k<=maximum;k++){
   if(vote(neighbors.slice(0,k)).pred===p.label)validHits[k-1]++;
   if(vote(trainNeighbors.slice(0,k)).pred===p.label)trainHits[k-1]++;
  }
 }
 const scores=validHits.map(n=>n/points.length);
 const training=trainHits.map(n=>n/points.length);
 let best=0;
 for(let i=1;i<scores.length;i++)if(scores[i]>scores[best]+1e-10)best=i;
 looCache={key,scores,training,best:best+1};
 return looCache;
}
function renderValidation(){
 const cv=$('validation-plot'),{ctx,w,h}=resizeCanvas(cv),data=looScores();
 const s=data.scores,train=data.training,maxK=s.length;
 ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
 const l=49,r=18,t=24,b=47,bot=h-b;
 // Keep at least 50%–100% visible, avoiding misleading almost-zero y-ranges.
 const minimum=Math.min(...s,...train);
 const lo=Math.min(.50,Math.max(0,Math.floor((minimum-.05)*10)/10));
 const hi=1;
 const X=k=>l+(k-1)/Math.max(1,maxK-1)*(w-l-r);
 const Y=a=>bot-(a-lo)/(hi-lo)*(bot-t);
 ctx.font='11px system-ui,sans-serif';ctx.textAlign='right';
 for(let i=0;i<=5;i++){
  const v=lo+(hi-lo)*i/5;
  ctx.strokeStyle='#d9e4e8';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(l,Y(v));ctx.lineTo(w-r,Y(v));ctx.stroke();
  ctx.fillStyle='#536879';ctx.fillText(Math.round(v*100)+'%',l-8,Y(v)+4);
 }
 ctx.textAlign='center';ctx.fillStyle='#536879';
 for(const k of [1,5,10,15,20,25].filter(v=>v<=maxK))ctx.fillText(String(k),X(k),bot+18);
 ctx.fillText('K (number of neighbours) · Click a K to try it',w/2,h-7);
 function curve(values,color){
  ctx.beginPath();
  values.forEach((v,i)=>i?ctx.lineTo(X(i+1),Y(v)):ctx.moveTo(X(i+1),Y(v)));
  ctx.strokeStyle=color;ctx.lineWidth=2.8;ctx.stroke();
  ctx.fillStyle=color;
  values.forEach((v,i)=>{ctx.beginPath();ctx.arc(X(i+1),Y(v),2.5,0,Math.PI*2);ctx.fill();});
 }
 curve(train,'#167b9a');curve(s,'#e26b54');
 if(state.k<=maxK){
  ctx.strokeStyle='#697a8b';ctx.lineWidth=1.5;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(X(state.k),t);ctx.lineTo(X(state.k),bot);ctx.stroke();ctx.setLineDash([]);
 }
 ctx.beginPath();ctx.arc(X(data.best),Y(s[data.best-1]),6.5,0,Math.PI*2);
 ctx.fillStyle='white';ctx.fill();ctx.strokeStyle='#257b4b';ctx.lineWidth=3;ctx.stroke();
 $('best-k').textContent=data.best;
 $('best-score').textContent=fmt(s[data.best-1]*100,1)+'%';
 $('current-score').textContent=state.k<=maxK?fmt(s[state.k-1]*100,1)+'%':'Choose K ≤ '+maxK+' to compare';
 $('cv-reading').textContent='With K = 1, training accuracy is '+fmt(train[0]*100,0)+'% while leave-one-out accuracy is '+fmt(s[0]*100,0)+'%. The highest observed leave-one-out accuracy is '+fmt(s[data.best-1]*100,0)+'% at K = '+data.best+'. A high training score alone does not establish generalization.';
 $('noise-output').textContent=state.labelNoise+'%';
}
// Chart interaction: choose K by clicking the validation curve.
$('validation-plot').addEventListener('click',e=>{
 const r=e.currentTarget.getBoundingClientRect();
 const l=49,right=18,maxK=looScores().scores.length;
 const k=clamp(Math.round(1+(e.clientX-r.left-l)/Math.max(1,r.width-l-right)*(maxK-1)),1,maxK);
 state.k=k;$('k').value=k;redrawCurrent();
});

function updateCode(){
 const c=CONFIG[type],standard=state.scaling==='standard',metric=state.metric,weights=state.weights;
 const lines=[
 '# 1. Export knn_data.csv from the interactive app first.',
 'import pandas as pd',
 'import matplotlib.pyplot as plt',
 'from sklearn.model_selection import train_test_split',
 'from sklearn.neighbors import KNeighborsClassifier',
 'from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay',
 standard?'from sklearn.preprocessing import StandardScaler':'',
 standard?'from sklearn.pipeline import make_pipeline':'',
 '',
 "df = pd.read_csv('knn_data.csv')",
 "X = df[['feature_1', 'feature_2']]  # "+c.xlabel+"; "+c.ylabel,
 "y = df['class_id']  # 0 or 1",
 '',
 '# 2. Hold out unseen examples. Stratify to preserve class balance.',
 'X_train, X_test, y_train, y_test = train_test_split(',
 '    X, y, test_size=0.25, random_state=42, stratify=y)',
 '',
 '# 3. Choose neighbours, distance and voting.',
 'knn = KNeighborsClassifier(',
 '    n_neighbors=min('+state.k+', len(X_train)),',
 "    metric='"+metric+"', weights='"+weights+"')",
 standard?'model = make_pipeline(StandardScaler(), knn)  # fit scaling on training data only':'model = knn  # raw input units; try standard scaling',
 'model.fit(X_train, y_train)',
 '',
 '# 4. Predict unseen examples, not just training rows.',
 'predictions = model.predict(X_test)',
 "print('Test accuracy:', round(accuracy_score(y_test, predictions), 3))",
 'ConfusionMatrixDisplay.from_predictions(y_test, predictions)',
 "plt.title('Unseen test examples'); plt.show()",
 '',
 '# 5. Predict the point you placed on the graph.',
 "unknown = pd.DataFrame([["+fmt(query[0],3)+", "+fmt(query[1],3)+"],",
 "                       columns=['feature_1', 'feature_2'])",
 'print("Prediction for unknown point:", model.predict(unknown)[0])',
 "# Classes: 0 = "+c.classes[0]+"; 1 = "+c.classes[1],
 ];
 $('python-code').textContent=lines.filter((s,i)=>s!==''||lines[i-1]!=='').join('\n');
}
function exportCSV(){
 const rows=['feature_1,feature_2,class_id'].concat(points.map(p=>[p.x,p.y,p.label].join(',')));
 const blob=new Blob([rows.join('\n')+'\n'],{type:'text/csv;charset=utf-8'});
 const a=document.createElement('a'),url=URL.createObjectURL(blob);
 a.href=url;a.download='knn_data.csv';document.body.appendChild(a);a.click();a.remove();
 setTimeout(()=>URL.revokeObjectURL(url),2000);
}
function update(full=true){
 if(full){boundaryCache=null;looCache=null;reveal=0;}
 render();renderValidation();updateCode();
 $('sample-count').textContent=points.length;
}
function reset(which=type){
 type=which;points=dataset(type);serial=points.length;version++;
 query=CONFIG[type].query.slice();state.k=5;reveal=0;state.mode='query';
 $('example').value=type;$('k').max=points.length;$('k').value=5;
 $('mouse-mode').value='query';$('x-name').textContent=CONFIG[type].xlabel;$('y-name').textContent=CONFIG[type].ylabel;
 $('legend-a').textContent=CONFIG[type].classes[0];$('legend-b').textContent=CONFIG[type].classes[1];
 $('clinical-note').hidden=type!=='patients';
 $('noise').value=state.labelNoise;
 update();
}
function redrawCurrent(){
 reveal=0;boundaryCache=null;render();renderValidation();updateCode();
}
function alterQuery(x,y){
 const [xmin,xmax,ymin,ymax]=CONFIG[type].range;
 query=[clamp(x,xmin,xmax),clamp(y,ymin,ymax)];
 reveal=0;render();updateCode();
}
$('example').addEventListener('change',e=>reset(e.target.value));
$('noise').addEventListener('input',e=>{state.labelNoise=Number(e.target.value);points=dataset(type);serial=points.length;version++;update();});
$('k').addEventListener('input',e=>{state.k=+e.target.value;redrawCurrent();});
$('metric').addEventListener('change',e=>{state.metric=e.target.value;update();});
$('scaling').addEventListener('change',e=>{state.scaling=e.target.value;update();});
$('weights').addEventListener('change',e=>{state.weights=e.target.value;update();});
$('show-boundary').addEventListener('change',e=>{state.boundary=e.target.checked;render();});
$('show-neighbours').addEventListener('change',e=>{state.rings=e.target.checked;render();});
$('mouse-mode').addEventListener('change',e=>{state.mode=e.target.value;render();});
$('reset-data').addEventListener('click',()=>reset());
$('next-neighbour').addEventListener('click',()=>{reveal=reveal>=state.k?1:reveal+1;render();});
$('best-apply').addEventListener('click',()=>{state.k=looScores().best;$('k').value=state.k;redrawCurrent();document.getElementById('lab').scrollIntoView({behavior:'smooth'});});
$('export-csv').addEventListener('click',exportCSV);
$('copy-python').addEventListener('click',async()=>{
 const b=$('copy-python');
 try{await navigator.clipboard.writeText($('python-code').textContent);b.textContent='Copied to clipboard';}
 catch{b.textContent='Select and copy the code below';}
 setTimeout(()=>{b.textContent='Copy Python code';},2500);
});
$('point-x').addEventListener('change',()=>alterQuery(Number($('point-x').value),query[1]));
$('point-y').addEventListener('change',()=>alterQuery(query[0],Number($('point-y').value)));
const plot=$('plot');
function position(e){const rect=plot.getBoundingClientRect(),g=geometry(rect.width,rect.height);return [g.unX(e.clientX-rect.left),g.unY(e.clientY-rect.top)];}
function interact(e){
 const [x,y]=position(e),[xmin,xmax,ymin,ymax]=CONFIG[type].range;
 if(state.mode!=='query'){
  if(x< xmin||x>xmax||y<ymin||y>ymax)return;
  points.push({x,y,label:state.mode==='add0'?0:1,id:serial++});version++;
  $('k').max=points.length;update();return;
 }
 alterQuery(x,y);
}
plot.addEventListener('pointerdown',e=>{if(e.button!==0&&e.pointerType==='mouse')return;dragging=state.mode==='query';if(dragging)plot.setPointerCapture(e.pointerId);interact(e);});
plot.addEventListener('pointermove',e=>{if(dragging)interact(e);});
plot.addEventListener('pointerup',()=>dragging=false);
plot.addEventListener('pointercancel',()=>dragging=false);
window.addEventListener('resize',()=>{boundaryCache=null;render();renderValidation();});
reset('flowers');

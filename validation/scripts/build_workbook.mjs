import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const folder=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const require=createRequire(path.join(folder,'.work','package.json'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const read=async p=>JSON.parse(await fs.readFile(path.join(folder,p),'utf8'));
const names=(await fs.readdir(path.join(folder,'specs'))).filter(x=>x.endsWith('.json')).sort();
const originals=await Promise.all(names.map(x=>read('specs/'+x)));
const specs=await Promise.all(names.map(x=>read('compact_specs/'+x)));
specs.sort((a,b)=>a.title.localeCompare(b.title));
const originalBySlug=new Map(originals.map(x=>[x.slug,x]));
const wb=Workbook.create(),index=wb.worksheets.add('Validation Index'),layouts=[],reports=[];
const colors={ink:'#18334A',header:'#254C73',input:'#FFF5DC',formula:'#E5F0FC',muted:'#586777'};
const letter=n=>{let s='';for(n++;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;};
const put=(s,a,v)=>s.getRange(a).values=[[v]];
const fx=(s,a,v)=>s.getRange(a).formulas=[[v]];
function head(sh,r){sh.getRange(r).format={fill:colors.header,font:{name:'Arial',size:11,color:'#FFFFFF',bold:true},wrapText:true,verticalAlignment:'center',rowHeight:42};}
function words(text,width){let a=[],v='';for(const w of text.split(/\s+/)){if((v+' '+w).length>width){a.push(v);v=w;}else v+=(v?' ':'')+w;}if(v)a.push(v);return a;}
function references(f,inputs,calcs){
  const ids=new Set([...f.matchAll(/@([A-Za-z0-9_]+)/g)].map(x=>x[1]));
  for(const m of f.matchAll(/@([A-Za-z0-9_]+):@([A-Za-z0-9_]+)/g)){
    const list=inputs.some(x=>x.id===m[1])?inputs:calcs;
    const lo=list.findIndex(x=>x.id===m[1]),hi=list.findIndex(x=>x.id===m[2]);
    if(lo<0||hi<lo)throw Error('Invalid range '+m[0]);
    for(const x of list.slice(lo,hi+1))ids.add(x.id);
  }return ids;
}
for(const compact of specs){
  const original=originalBySlug.get(compact.slug);
  const inputs=[...original.inputs,...(compact.extraInputs||[])],calcs=[...original.calculations,...(compact.extraCalculations||[])];
  const byId=new Map([...inputs,...calcs].map(x=>[x.id,x])),calcById=new Map(calcs.map(x=>[x.id,x]));
  const selected=compact.selectedCheckIndices.map(i=>({...original.checks[i],index:i,formula:compact.formulaOverrides?.[i]||original.checks[i].formula}));
  for(const c of selected){let n=0;while(/^=@\w+$/.test(c.formula)&&calcById.has(c.formula.slice(2))){if(n++>30)throw Error('Cyclic alias');c.formula=calcById.get(c.formula.slice(2)).formula;}}
  const required=new Set(),visit=f=>{for(const id of references(f,inputs,calcs)){if(required.has(id))continue;required.add(id);const x=byId.get(id);if(!x)throw Error(compact.slug+': missing '+id);if(x.formula)visit(x.formula);}};
  selected.forEach(x=>visit(x.formula));
  compact.rawRows.flat().filter(x=>x&&typeof x==='object'&&x.formula).forEach(x=>visit(x.formula));
  let helpers=calcs.filter(x=>required.has(x.id));
  const sh=wb.worksheets.add(compact.title),rawCols=compact.rawHeaders.length;
  const labelCol=rawCols+1,excelCol=labelCol+1,actualCol=labelCol+2,errorCol=labelCol+3;
  const phaseChecks=selected.filter(c=>/phase/i.test(c.metric)&&['deg','rad'].includes(c.unit));
  const phaseCol=phaseChecks.length?errorCol+1:null,lastCol=phaseCol??errorCol;
  const A=letter(labelCol),E=letter(excelCol),G=letter(actualCol),P=letter(errorCol),lastLetter=letter(lastCol);
  const first=8,end=first+selected.length-1,helperStart=end+4,map=new Map();
  compact.rawRows.forEach((row,r)=>{if(row.length!==rawCols)throw Error(compact.slug+': uneven raw row');row.forEach((v,c)=>{
    if(v&&typeof v==='object'&&v.id){
      if(map.has(v.id))throw Error(compact.slug+': duplicate raw id '+v.id);
      const authored=byId.get(v.id);
      if(!v.formula&&authored&&JSON.stringify(v.value)!==JSON.stringify(authored.value))throw Error(compact.slug+': changed raw '+v.id);
      map.set(v.id,`$${letter(c)}$${first+r}`);
    }
  });});
  helpers=helpers.filter(x=>!map.has(x.id));
  const parameters=inputs.filter(x=>required.has(x.id)&&!map.has(x.id)),parameterStart=first+compact.rawRows.length+3;
  parameters.forEach((x,i)=>map.set(x.id,`$${letter(Math.max(rawCols-2,1))}$${parameterStart+i}`));
  helpers.forEach((x,i)=>map.set(x.id,`$${E}$${helperStart+i}`));
  const resolve=f=>f.replace(/@([A-Za-z0-9_]+)/g,(_,id)=>{if(!map.has(id))throw Error(compact.slug+': unmapped '+id);return map.get(id);});
  for(const f of [...selected,...helpers].map(x=>x.formula))for(const m of f.matchAll(/@([A-Za-z0-9_]+):@([A-Za-z0-9_]+)/g)){
    const list=inputs.some(x=>x.id===m[1])?inputs:calcs,lo=list.findIndex(x=>x.id===m[1]),hi=list.findIndex(x=>x.id===m[2]);
    const parsed=list.slice(lo,hi+1).map(x=>map.get(x.id)?.match(/^\$([A-Z]+)\$(\d+)$/));
    if(parsed.some(x=>!x)||parsed.some((x,i)=>x[1]!==parsed[0][1]||+x[2]!==+parsed[0][2]+i))throw Error(compact.slug+': noncontiguous '+m[0]);
  }
  const last=Math.max(end,helperStart+helpers.length,parameterStart+parameters.length,first+compact.rawRows.length)+6;
  sh.showGridLines=false;sh.getRange(`A1:${lastLetter}${last}`).format={font:{name:'Arial',size:11,color:colors.ink},rowHeight:27,verticalAlignment:'center'};
  compact.rawHeaders.forEach((h,i)=>{const w=/dataset|source/i.test(h)?12:/polar|^pol$/i.test(h)?7:/frequency/i.test(h)?12:/phase|rcs|power/i.test(h)?14:/sqrt/i.test(h)?12:11;sh.getRange(`${letter(i)}1:${letter(i)}${last}`).format.columnWidth=w;});
  sh.getRange(`${letter(rawCols)}1:${letter(rawCols)}${last}`).format.columnWidth=3;
  for(const [c,w] of [[A,31],[E,17],[G,17],[P,15],...(phaseCol!==null?[[letter(phaseCol),14]]:[])])sh.getRange(`${c}1:${c}${last}`).format.columnWidth=w;
  put(sh,'A1',compact.title+(original.category.startsWith('BETA')?' (BETA)':''));sh.getRange('A1').format.font={name:'Arial',size:18,bold:true,color:colors.ink};sh.getRange(`A1:${lastLetter}1`).format.rowHeight=35;
  const setup=words(compact.settings,150);put(sh,'A3',setup[0]);if(setup.length>1)put(sh,'A4',setup.slice(1).join(' '));
  put(sh,'A5',`Blue cells ${E}${first}:${E}${end} contain the Excel formulas. Click a blue cell to see its formula.`);sh.getRange(`A5:${lastLetter}5`).format.font={name:'Arial',size:10,color:colors.muted};
  put(sh,'A6','Raw data');put(sh,`${A}6`,'Calculation and comparison');sh.getRange(`A6:${lastLetter}6`).format.font={name:'Arial',size:11,bold:true,color:colors.ink};
  sh.getRange(`A7:${letter(rawCols-1)}7`).values=[compact.rawHeaders.map(x=>x==='Polarization'?'Pol':x)];head(sh,`A7:${letter(rawCols-1)}7`);
  sh.getRange(`${A}7:${lastLetter}7`).values=[['Quantity','Excel calculation','GRIM exported','Percent error',...(phaseCol!==null?['Phase error (deg)']:[])]];head(sh,`${A}7:${lastLetter}7`);
  const rawFormulaCells=[];
  compact.rawRows.forEach((row,r)=>row.forEach((v,c)=>{const a=`${letter(c)}${first+r}`;
    if(v&&typeof v==='object'&&v.formula){fx(sh,a,resolve(v.formula));rawFormulaCells.push(a);sh.getRange(a).format.fill=colors.formula;}
    else{put(sh,a,v&&typeof v==='object'?v.value:v);sh.getRange(a).format.fill=colors.input;}
    sh.getRange(a).setNumberFormat(v&&typeof v==='object'&&v.formula?'0.000000':'General');const n=v&&typeof v==='object'?v.value:v;
    if(typeof n==='number'&&!Number.isInteger(n))sh.getRange(a).setNumberFormat('0.000000');
    sh.getRange(a).format.horizontalAlignment=/polar|^pol$/i.test(compact.rawHeaders[c])?'center':typeof n==='string'&&n!=='missing'?'left':'right';
    if(typeof n==='string'&&n!=='missing')sh.getRange(a).format.wrapText=true;
    if(typeof n==='number'&&n!==0&&(Math.abs(n)<1e-4||Math.abs(n)>=1e6))sh.getRange(a).setNumberFormat('0.0000E+00');
  }));
  if(parameters.length){put(sh,`A${parameterStart-1}`,'Settings used in formulas');sh.getRange(`A${parameterStart-1}`).format.font={name:'Arial',size:11,bold:true};}
  parameters.forEach((x,i)=>{put(sh,`A${parameterStart+i}`,x.label+(x.unit?' ('+x.unit+')':''));put(sh,map.get(x.id),x.value);sh.getRange(map.get(x.id)).format.fill=colors.input;sh.getRange(map.get(x.id)).setNumberFormat('General');});
  const checks=[];
  selected.forEach((c,i)=>{const r=first+i,e=`${E}${r}`,g=`${G}${r}`,p=`${P}${r}`;
    const simpleCase=c.case.replace(/^result sample /i,'Output ').replace(/^result_/,'').replace(/ sample /i,' ');
    const simpleMetric=c.metric.replace('magnitude_power_linear','RCS').replace(/^Linear power$/i,'RCS');
    const simpleUnit=c.unit==='linear power'?'m²':c.unit.replace('m^2','m²').replace(/^m2$/,'m²');
    const label=compact.labelOverrides?.[c.index]||`${simpleCase}: ${simpleMetric}${simpleUnit?' ('+simpleUnit+')':''}`;
    put(sh,`${A}${r}`,label);fx(sh,e,resolve(c.formula));put(sh,g,c.actual===null?'missing':c.actual);sh.getRange(e).format.fill=colors.formula;sh.getRange(`${A}${r}`).format.wrapText=true;
    sh.getRange(`A${r}:${lastLetter}${r}`).format.rowHeight=Math.max(32,16*Math.ceil(label.length/32));
    const phase=/phase/i.test(c.metric)&&['deg','rad'].includes(c.unit),db=/^db(sm|ke)?$/i.test(c.unit)&&compact.slug!=='rf_compare';
    const numeric=`AND(ISNUMBER(${e}),ISNUMBER(${g}))`,missing=`IF(ISBLANK(${g}),"missing export",IF(ISBLANK(${e}),"missing input","n/a"))`;
    const percent=phase?'"n/a (angle)"':db?`ABS(10^((${g}-${e})/10)-1)`:`IF(SIGN(${e})=0,"n/a (zero)",ABS(${g}-${e})/ABS(${e}))`;
    fx(sh,p,`=IF(${numeric},${percent},${missing})`);sh.getRange(p).setNumberFormat('0.0000%');sh.getRange(p).format.horizontalAlignment='right';
    let phaseErrorCell=null;if(phase){phaseErrorCell=`${letter(phaseCol)}${r}`;const factor=c.unit==='rad'?'*180/PI()':'';fx(sh,phaseErrorCell,`=IF(${numeric},ABS(MOD((${g}-${e})${factor}+180,360)-180),"n/a")`);sh.getRange(phaseErrorCell).setNumberFormat('0.000000');sh.getRange(phaseErrorCell).format.horizontalAlignment='right';}
    sh.getRange(`${e}:${g}`).setNumberFormat('0.000000');if(typeof c.actual==='number'&&c.actual!==0&&(Math.abs(c.actual)<1e-4||Math.abs(c.actual)>=1e6))sh.getRange(`${e}:${g}`).setNumberFormat('0.0000E+00');
    checks.push({row:r,label,expectedCell:e,actualCell:g,errorCell:p,phaseErrorCell,unit:c.unit,tolerance:c.tolerance??null,atol:original.atol,rtol:original.rtol,originalIndex:c.index,source:c.source,errorMode:phase?'phase':db?'db':'relative'});
  });
  if(helpers.length){put(sh,`${A}${helperStart-1}`,'Calculation steps');sh.getRange(`${A}${helperStart-1}`).format.font={name:'Arial',size:11,bold:true};}
  helpers.forEach((x,i)=>{const r=helperStart+i,label=x.label+(x.unit?' ('+x.unit+')':'');put(sh,`${A}${r}`,label);fx(sh,`${E}${r}`,resolve(x.formula));sh.getRange(`${E}${r}`).format.fill=colors.formula;sh.getRange(`${E}${r}`).setNumberFormat('0.000000');sh.getRange(`${A}${r}`).format.wrapText=true;sh.getRange(`A${r}:${lastLetter}${r}`).format.rowHeight=Math.max(32,16*Math.ceil(label.length/32));});
  compact.rawRows.forEach((row,i)=>{const longest=Math.max(...row.map((v,c)=>{const n=v&&typeof v==='object'?v.value:v;return typeof n==='string'&&n!=='missing'?Math.ceil(n.length/(compact.rawHeaders[c].match(/dataset|source/i)?12:13)):1;}));if(longest>1)sh.getRange(`A${first+i}:${lastLetter}${first+i}`).format.rowHeight=Math.max(32,16*longest,16*Math.ceil((checks[i]?.label.length||0)/32));});
  sh.getRange(`A8:${letter(rawCols-1)}${first+compact.rawRows.length-1}`).format.borders={insideVertical:{style:'thin',color:'#FFFFFF'}};
  const noteRow=Math.max(first+compact.rawRows.length,parameterStart+parameters.length)+2;put(sh,`A${noteRow}`,'Files: datasets/'+compact.slug+'/ (see GUIDE.txt)');
  put(sh,`A${last-2}`,'Percent error = |GRIM − Excel| / |Excel|. A zero reference is shown as n/a.');
  if(selected.some(c=>/^db(sm|ke)?$/i.test(c.unit))&&compact.slug!=='rf_compare')put(sh,`A${last-1}`,'For dB results, percent error compares the equivalent linear power.');
  if(phaseCol!==null)put(sh,`A${last}`,'Phase is compared using wrapped error in degrees; percent error is not defined for angles.');
  sh.getRange(`A${last-2}:${lastLetter}${last}`).format.font={name:'Arial',size:10,color:colors.muted};sh.freezePanes.freezeRows(7);
  layouts.push({slug:compact.slug,title:compact.title,checks,checkCount:checks.length,firstExpected:checks[0].expectedCell,firstActual:checks[0].actualCell,inputCells:Object.fromEntries(map),helperCount:helpers.length,helperCells:helpers.map((x,i)=>`${E}${helperStart+i}`),parameterCount:parameters.length,last,lastColumn:lastLetter,rawFormulaCells});
}
index.showGridLines=false;index.getRange('A1:D60').format={font:{name:'Arial',size:11,color:colors.ink},rowHeight:25};
for(const [c,w] of Object.entries({A:28,B:38,C:20,D:48}))index.getRange(`${c}1:${c}60`).format.columnWidth=w;
put(index,'A1','GRIM validation examples');index.getRange('A1').format.font={name:'Arial',size:18,bold:true};index.getRange('A1:D1').format.rowHeight=35;
put(index,'A3','Each sheet puts raw data beside Excel calculations, GRIM exports and percent error.');put(index,'A4','Blue cells contain live formulas. Yellow cells hold raw inputs and settings.');put(index,'A5','Compact examples are shown here. Full datasets remain in the folders.');
index.getRange('A7:C7').values=[['Validation','Dataset folder','Comparisons']];head(index,'A7:C7');
specs.forEach((s,i)=>index.getRange(`A${8+i}:C${8+i}`).values=[[s.title,'datasets/'+s.slug,layouts[i].checkCount]]);index.getRange('C8:C53').setNumberFormat('0');index.freezePanes.freezeRows(7);
wb.recalculate();
for(const l of layouts){const sh=wb.worksheets.getItem(l.title);for(const a of [...l.rawFormulaCells,...l.helperCells]){const n=sh.getRange(a).values[0][0];if(typeof n==='number'&&n!==0&&(Math.abs(n)<1e-4||Math.abs(n)>=1e6))sh.getRange(a).setNumberFormat('0.0000E+00');}}
for(const l of layouts){const sh=wb.worksheets.getItem(l.title),failures=[];for(const c of l.checks){const expected=sh.getRange(c.expectedCell).values[0][0],actual=sh.getRange(c.actualCell).values[0][0],tol=c.tolerance??c.atol+c.rtol*Math.abs(expected);const valid=typeof expected==='number'&&typeof actual==='number'?Math.abs(actual-expected)<=tol:expected===actual;if(!valid)failures.push({...c,expected,actual,tolerance:tol});}reports.push({sheet:l.title,checks:l.checkCount,failures});}
await fs.writeFile(path.join(folder,'.work','artifact_check_results.json'),JSON.stringify(reports,null,2));await fs.writeFile(path.join(folder,'.work','workbook_layout.json'),JSON.stringify(layouts,null,2));
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},summary:'Formula error scan',maxChars:2000})).ndjson);
console.log((await wb.inspect({kind:'table',range:'Stats!H7:K13',include:'values,formulas',tableMaxRows:7,tableMaxCols:4,maxChars:2500})).ndjson);
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(folder,'GRIM_validation_simple.xlsx'));
console.log(JSON.stringify({sheets:specs.length,comparisons:reports.reduce((n,r)=>n+r.checks,0),failures:reports.reduce((n,r)=>n+r.failures.length,0),largestHelperCount:Math.max(...layouts.map(x=>x.helperCount))}));
if(process.argv.includes('--render')){await fs.mkdir(path.join(folder,'.work','simple_previews'),{recursive:true});for(const l of [{title:'Validation Index',slug:'index',lastColumn:'C',last:23},...layouts]){const blob=await wb.render({sheetName:l.title,range:`A1:${l.lastColumn}${Math.min(l.last,27)}`,scale:1,format:'png'});await fs.writeFile(path.join(folder,'.work','simple_previews',l.slug+'.png'),new Uint8Array(await blob.arrayBuffer()));}}

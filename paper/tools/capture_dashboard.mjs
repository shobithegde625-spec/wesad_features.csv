// Capture the real running Streamlit application through Chrome's local CDP.
// Node >=22; no npm packages. Chrome must expose port 9224; Streamlit port 8502.
import fs from 'node:fs/promises';
import path from 'node:path';
const out = path.resolve('paper/figures');
await fs.mkdir(out, {recursive:true});
const tabs = await (await fetch('http://127.0.0.1:9224/json/list')).json();
const tab = tabs.find(t=>t.type==='page');
if (!tab) throw Error('No Chrome page target');
const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
let sequence=0;
const pending=new Map();
ws.onmessage=e=>{const r=JSON.parse(e.data); if(pending.has(r.id)) {
  const {resolve,reject,timer}=pending.get(r.id); clearTimeout(timer);pending.delete(r.id);
  r.error?reject(Error(JSON.stringify(r.error))):resolve(r.result);
}};
function command(method,params={}) {return new Promise((resolve,reject)=>{
  const id=++sequence; const timer=setTimeout(()=>{pending.delete(id);reject(Error(method+' timed out'));},120000);
  pending.set(id,{resolve,reject,timer}); ws.send(JSON.stringify({id,method,params}));
});}
async function evaluate(expression) {
  const r=await command('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
  if(r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
}
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function waitFor(expression,timeout=90000) {
  const end=Date.now()+timeout;
  while(Date.now()<end) {if(await evaluate(expression)) return;await pause(500);}
  throw Error('UI did not reach expected state: '+expression);
}
async function page(name) {
  await evaluate(`(()=>{const r=[...document.querySelectorAll('[role="radiogroup"] label')].find(x=>x.innerText.trim()===${JSON.stringify(name)});if(!r)throw Error('Page control missing');r.click();})()`);
  const titles={Overview:'Stress detection workspace',Prediction:'Run a prediction',Monitoring:'Monitoring replay',Analysis:'Evaluation analysis',Performance:'Model performance'};
  await waitFor(`document.body.innerText.includes(${JSON.stringify(titles[name])})`);
  await pause(1400);
}
async function clickButton(text) {
  await evaluate(`(()=>{const b=[...document.querySelectorAll('button')].find(x=>x.innerText.trim()===${JSON.stringify(text)});if(!b)throw Error('Button missing');b.click();})()`);
}
async function screenshot(name) {
  // Hide local filesystem fields in the screenshot only; keep app behavior intact.
  await evaluate(`(()=>{let s=document.getElementById('paper-capture-style');if(!s){s=document.createElement('style');s.id='paper-capture-style';document.head.appendChild(s);}s.textContent='[data-testid="stSidebar"] [data-testid="stTextInput"]{display:none!important}[data-testid="stSidebar"] [data-baseweb="select"] *{color:#183345!important}';window.scrollTo(0,0);const main=document.querySelector('[data-testid="stMain"]');if(main)main.scrollTop=0;})()`);
  const r=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
  await fs.writeFile(path.join(out,name),Buffer.from(r.data,'base64'));
  console.log('Captured '+name);
}
try {
  await command('Page.enable');
  await command('Runtime.enable');
  await command('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:'light'}]});
  await command('Emulation.setDeviceMetricsOverride',{width:1500,height:1060,deviceScaleFactor:1,mobile:false});
  await command('Page.navigate',{url:'http://127.0.0.1:8502'});
  await waitFor(`document.body.innerText.includes('Stress detection workspace')`);
  await pause(2000);
  await screenshot('dashboard_overview.png');
  await page('Prediction');
  await clickButton('Run Prediction');
  await waitFor(`document.body.innerText.includes('Processing time') || document.body.innerText.includes('Selected expert') || document.body.innerText.includes('accepted')`,90000);
  await pause(5000);
  await screenshot('dashboard_prediction.png');
  await page('Monitoring');
  await clickButton('Start / Resume');
  await pause(12000);
  await clickButton('Stop');
  await pause(2000);
  await screenshot('dashboard_monitoring.png');
  await page('Performance');
  await screenshot('dashboard_provenance.png');
  await fs.writeFile(path.join(out,'screenshots.json'),JSON.stringify({
    captured_at:new Date().toISOString(),source:'Actual local Streamlit app.py',
    artifact_id:'dae3a7c563c9',scenario:'clean',policy:'cascade',
    note:'Light browser theme; local filesystem fields hidden and sidebar select-text contrast corrected with capture-only CSS. No predictions or metrics edited.'
  },null,2)+'\n');
} finally {ws.close();}

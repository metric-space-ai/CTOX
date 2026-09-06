// Component browser acceptance: real renderer and shell CSS, synthetic task data.
// Production authentication, replication and performance are separate gates.
import assert from 'node:assert/strict';
import http from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { stampShellDocument } from '../../../scripts/build-shell-artifact.mjs';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const out = path.resolve(root, '../../../output/welsch-harness-layout-20260906');
await mkdir(out, { recursive: true });
const bundle = await build({ entryPoints: [path.join(root, 'modules/ctox/index.js')], bundle: true, write: false, format: 'esm', platform: 'browser', logLevel: 'silent' });
const markup = await readFile(path.join(root, 'modules/ctox/index.html'), 'utf8');
const html = stampShellDocument(Buffer.from(`<!doctype html><html data-theme="dark"><head>
<link rel="stylesheet" href="/app.css"><link rel="stylesheet" href="/shared/base.css">
<link rel="stylesheet" href="/modules/ctox/index.css"></head><body>
<div class="shell-window-layer" style="position:fixed;inset:0"><section class="shell-window is-focused" data-shell-window="true" data-shell-contract="v2" data-shell-window-chrome="shared-v2" data-shell-header-rows="2" data-shell-icon-rows="2" data-owner-id="desktop-app:ctox" style="position:absolute;inset:0;width:100%;height:100%">
<div class="shell-window-content"><div class="module-root shell-window-module-root" data-module-root="ctox" data-module-ready="true"><aside class="shell-window-module-pane shell-window-module-pane--left"></aside><div class="shell-window-module-column-resizer shell-window-module-column-resizer--left"></div><main class="module-content" data-module-content>${markup}</main><div class="shell-window-module-column-resizer shell-window-module-column-resizer--right"></div><aside class="shell-window-module-pane shell-window-module-pane--right"></aside></div></div>
</section></div><script type="module">
import {__ctoxTestHooks as hooks} from '/bundle.js';
import {readEmbeddedIdentity} from '/shared/shell-release-status.js';
const host=document.querySelector('[data-module-root]');
const task={id:'layout-task',title:'Verify the complete source and retain the measured acceptance evidence',status:'failed',executionPhase:'terminal',terminalStatus:'failed',executionProgress:{phase:'failed',percent:90,currentStep:3,completedSteps:3,totalSteps:3,steps:[1,2,3].map(position=>({position,label:'Verify source and retained evidence for this task',status:'completed',activityTurns:2})),activityTurns:{total:6,thinking:2,tools:4},updatedAtMs:1720000000123}};
const model=hooks.buildHarnessModel({runs:[],queue:[],communications:[],tickets:[],tools:[]},{ok:false},'de');
model.tasks=[task]; model.activeTask=task; model.activeNodeId='model-failed';
const state={ctx:{host},model,lang:'de',flow:{ok:false},selectedTaskId:task.id,selectedStepIndex:0,selectedTaskStepIndex:2,selectedNodeId:'',zoom:1,taskSearch:'',taskViewMode:'cards',taskPrimaryView:'all',taskSourceFilter:'all',taskPinFilter:'all',taskSort:'updated',taskSortDirection:'desc',pinnedTaskIds:new Set(),webStackPanelOpen:false,webStack:{loading:false,data:null,error:''},dataLoaded:true,dataError:'',runtimeStatus:'ready',flowViewport:{left:0,top:0}};
host.querySelector('[data-ctox-left]').innerHTML=hooks.taskColumnMarkup(model.tasks,state);
hooks.renderMain(state);
document.body.dataset.loadedVersion=readEmbeddedIdentity(document).version;
document.body.dataset.fixtureReady='true';
</script></body></html>`), { version: '1.2.3-beta.1', sourceCommit: 'a'.repeat(40) });
const server = http.createServer(async (req,res) => {
  try {
    const pathname=new URL(req.url,'http://localhost').pathname;
    if(pathname==='/'){res.setHeader('content-type','text/html');res.end(html);return;}
    if(pathname==='/bundle.js'){res.setHeader('content-type','text/javascript');res.end(bundle.outputFiles[0].contents);return;}
    const file=path.resolve(root, '.'+pathname);
    if(!file.startsWith(root)){res.writeHead(403).end();return;}
    res.setHeader('content-type',file.endsWith('.css')?'text/css':file.endsWith('.js')?'text/javascript':'application/octet-stream');
    res.end(await readFile(file));
  }catch{res.writeHead(404).end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({headless:true});
const results=[];
try {
  for(const width of [430,630,768,1000,1280]){
    const page=await browser.newPage({viewport:{width,height:710}});
    const errors=[];
    const requests=[];
    page.on('pageerror',error=>errors.push(String(error)));
    page.on('request',request=>requests.push(new URL(request.url()).pathname));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.locator('body[data-fixture-ready="true"]').waitFor({timeout:10000}).catch(error=>{throw new Error(errors.join('\n')||String(error));});
    const measured=await page.locator('[data-flow-canvas]').evaluate(canvas=>({height:canvas.getBoundingClientRect().height,width:canvas.getBoundingClientRect().width,nodes:canvas.querySelectorAll('.ctox-flow-node-g').length}));
    results.push({viewportWidth:width,...measured,errors});
    await page.screenshot({path:path.join(out,`${width}.png`)});
    assert.equal(errors.length,0);
    assert.equal(await page.locator('body').getAttribute('data-loaded-version'),'1.2.3-beta.1');
    assert.ok(!requests.includes('/ctox-shell-manifest.json'),'loaded identity must not require a second manifest request');
    assert.equal(measured.nodes,16);
    assert.ok(measured.height>=200,`Harness collapsed at width ${width}: ${JSON.stringify(measured)}`);
    await page.locator('[data-node-id="queued"]').scrollIntoViewIfNeeded();
    const visible=await page.locator('[data-node-id="queued"]').evaluate(node=>{const r=node.getBoundingClientRect();return document.elementsFromPoint(r.x+r.width/2,r.y+r.height/2).some(e=>e===node||node.contains(e));});
    assert.ok(visible,`Harness node is clipped at width ${width}`);
    await page.close();
  }
  console.log(JSON.stringify({passed:results.length,results}));
}finally{
  await writeFile(path.join(out,'results.json'),JSON.stringify(results,null,2));
  await browser.close();
  await new Promise(resolve=>server.close(resolve));
}

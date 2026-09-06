import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
const stage='/home/ctox/.cache/ctox/file-preservation-fbeca32b7';
const root=stage+'/browser-timing-release-final-20260906';
assert.equal(await fs.realpath(root),root);
assert.equal(await fs.realpath(root+'/runtime'),root+'/runtime');
await fs.access(root+'/src/core/main.rs');
await fs.access(root+'/contracts/history/creation-ledger.md');
const binary=stage+'/release-target/release/ctox';
assert.equal(createHash('sha256').update(await fs.readFile(binary)).digest('hex'),'580887fe5133089d77dc6dd1a44c0e865e539c9a1d22285738ce6ee8d602790c');
const version='0.1.46-beta.13';
const slot=root+'/runtime/business-os-shell/slots/'+version;
const css=slot+'/app.css';
const original=await fs.readFile(css);
const child=spawn(binary,['business-os','serve','--addr','127.0.0.1:28977'],{cwd:root,env:{...process.env,CTOX_ROOT:root},detached:true,stdio:['ignore','pipe','pipe']});
let output=''; child.stdout.on('data',d=>output+=d);child.stderr.on('data',d=>output+=d);
const exited=new Promise(resolve=>child.once('exit',(code,signal)=>resolve({code,signal})));
const timer=setTimeout(()=>{try{process.kill(-child.pid,'SIGKILL')}catch{}},120000);
const base='http://127.0.0.1:28977';
const results=[];
async function get(url,status){const started=performance.now();const response=await fetch(base+url,{signal:AbortSignal.timeout(30000)});const bytes=Buffer.from(await response.arrayBuffer());assert.equal(response.status,status,url+': '+bytes.toString().slice(0,150));results.push({url,status:response.status,bytes:bytes.length,elapsedMs:performance.now()-started});return {response,bytes};}
try {
 for(let i=0;i<100&&!output.includes('CTOX Business OS listening');i++)await new Promise(r=>setTimeout(r,100));
 assert(output.includes('CTOX Business OS listening'),output.slice(-1000));
 for (const entry of ['/', '/index.html', '/business-os', '/business-os/']) {
 const html=await get(entry,200);
 assert(html.bytes.toString().includes('<base href="/business-os/_shell/'+version+'/">'),entry);
 }
 const prefix='/business-os/_shell/'+version+'/';
 const first=await get(prefix+'app.css?v=unrelated-shell-v2-token',200);
 assert.deepEqual(first.bytes,original);
 assert.match(first.response.headers.get('cache-control'),/immutable/);
 await get('/business-os/_shell/9.9.999/app.js',410);
 await get(prefix+'%00app.js',400);
 await get(prefix+'does-not-exist.js',404);
 const corrupted=Buffer.from(original);corrupted[0]^=1;
 await fs.writeFile(css,corrupted);
 await get(prefix+'app.css',503);
 await fs.writeFile(css,original);
 const restored=await get(prefix+'app.css',200);assert.deepEqual(restored.bytes,original);
 await fs.rename(css,css+'.probe-backup');
 try{await get(prefix+'app.css',503)}finally{await fs.rename(css+'.probe-backup',css)}
 const last=await get(prefix+'app.css',200);assert.deepEqual(last.bytes,original);
 console.log(JSON.stringify({schema:'ctox.shell.release-http-probe.v1',binarySha256:'580887fe5133089d77dc6dd1a44c0e865e539c9a1d22285738ce6ee8d602790c',root,passed:true,latencyBudgetMs:5000,latencyBudgetPassed:results.every(r=>r.elapsedMs<5000),results},null,2));
 assert(results.every(r=>r.elapsedMs<5000),'one or more requests exceeded the retained 5-second latency guard');
} finally {
 await fs.writeFile(css,original);
 try{process.kill(-child.pid,'SIGTERM')}catch{}
 await exited;clearTimeout(timer);
 await fs.writeFile(stage+'/native-shell-http-probe-server.log',output);
}

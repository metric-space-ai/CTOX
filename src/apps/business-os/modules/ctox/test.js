import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {mergeBundleWithCommands, commandTaskFromProjection, normalizeExecutionProgress, queries, readCollection, mayControl, mayReadPrivate, taskGroup, taskActionReason} from './model.js';
import {AXES,translate,homeMarkup,workplaceMarkup,queueMarkup,profileMarkup} from './view.js';
import {canUseBusinessPermission,BusinessOsPermissions} from '../../shared/permissions.js';
import './tests/creature.test.mjs';
const de=JSON.parse(readFileSync(new URL('./locales/de.json',import.meta.url)));
const en=JSON.parse(readFileSync(new URL('./locales/en.json',import.meta.url)));
const actor=role=>({session:{user:{id:'operator',role}}});

// Preserved projection regressions from the retired SVG view: assertions target
// the same lifecycle evidence; the new workplace replaces map-node assertions.
const runningCommand={id:'command-runtime-1',command_id:'command-runtime-1',contract_version:2,module:'example-module',command_type:'example.work.execute',execution_mode:'queue',execution_task_id:'queue-runtime-1',execution_phase:'running',terminal_status:'none',projection_version:7,status:'accepted',payload:{title:'Execute example work'},updated_at_ms:100000};
test('Authoritative running command lifecycle overrides a stale queued task projection',()=>{
 const bundle=mergeBundleWithCommands({},[runningCommand],[{id:'queue-runtime-1',command_id:'command-runtime-1',title:'Execute example work',status:'queued',route_status:'pending',module:'example-module',updated_at_ms:70000,crew_member_id:'milo',hold_reason:'retained'}]);
 assert.equal(bundle.queue.length,1);assert.equal(bundle.queue[0].id,'queue-runtime-1');assert.equal(bundle.queue[0].commandId,'command-runtime-1');assert.equal(bundle.queue[0].status,'running');assert.equal(bundle.queue[0].routeStatus,'running');assert.equal(bundle.queue[0].executionPhase,'running');assert.equal(bundle.queue[0].crew_member_id,'milo');assert.equal(bundle.queue[0].hold_reason,'retained');
});
test('Running command remains active when legacy flow telemetry is stale',()=>{
 const {queue}=mergeBundleWithCommands({flow:{status:'completed'}},[runningCommand],[{id:'queue-runtime-1',status:'queued',route_status:'pending'}]);
 assert.equal(taskGroup(queue[0]),'running');assert.equal(queue.filter(t=>taskGroup(t)==='running').length,1);assert.equal(queue.filter(t=>taskGroup(t)==='waiting').length,0);
});
test('Running command without a queue projection is synthesized from its execution link',()=>{
 const {queue}=mergeBundleWithCommands({},[runningCommand],[]);assert.equal(queue.length,1);assert.equal(queue[0].id,'queue-runtime-1');assert.equal(queue[0].status,'running');assert.equal(taskGroup(queue[0]),'running');
});
test('Synchronous control commands do not become task overview items',()=>{
 const {queue}=mergeBundleWithCommands({},[{...runningCommand,execution_mode:'control',execution_phase:'terminal',terminal_status:'completed'}],[]);assert.deepEqual(queue,[]);
 assert.equal(commandTaskFromProjection({id:'unlinked',contract_version:2}),null);
});
test('Execution telemetry preserves zero, unknown, review-only state and step evidence',()=>{
 assert.equal(normalizeExecutionProgress(null),null);assert.equal(normalizeExecutionProgress({}),null);
 assert.equal(normalizeExecutionProgress({percent:0}).percent,0);assert.equal(normalizeExecutionProgress({percent:null}),null);
 assert.equal(normalizeExecutionProgress({review:{status:'validating'}}).reviewStatus,'validating');
 assert.deepEqual(normalizeExecutionProgress({steps:[{position:2,label:'Verify',status:'in_progress',activity_turns:3}]}).steps,[{position:2,label:'Verify',status:'in_progress',activityTurns:3}]);
});
test('Every query has selector and explicit bounded limit, event/run/learning reads are scoped',async()=>{
 const all=[queries.members(),queries.status(),queries.tasks(),queries.tasks({terminal:true,source:'ctox',search:'a.b'}),queries.task('t'),queries.commands(['c']),queries.activeCommands(),queries.events('t'),queries.runs({taskId:'t'}),queries.runs({memberId:'m'}),queries.learnings('m')];
 for(const query of all){assert.ok(Object.keys(query.selector).length);assert.ok(query.limit>0&&query.limit<=200);}
 assert.deepEqual(queries.events('t').selector,{task_id:{$eq:'t'}});assert.deepEqual(queries.runs({memberId:'m'}).selector,{crew_member_id:{$eq:'m'}});assert.deepEqual(queries.learnings('m').selector,{member_id:{$eq:'m'}});
 const cursor={updated_at_ms:123,id:'b'};assert.deepEqual(queries.tasks({cursor}).selector.$and[1],{$or:[{updated_at_ms:{$lt:123}},{$and:[{updated_at_ms:{$eq:123}},{id:{$lt:'b'}}]}]});
 await assert.rejects(readCollection({},'ctox_queue_tasks',{selector:{},limit:200}),/invalid_bounded_query/);
});
test('Role aliases, grants and owned task scope mirror server restrictions',()=>{
 for(const role of ['admin','chef','owner','business_os_admin']){assert.equal(mayControl(actor(role),'ctox.crew.member.update'),true);assert.equal(mayReadPrivate(actor(role)),true);}
 assert.equal(mayControl(actor('founder'),'ctox.crew.member.update'),false);assert.equal(mayControl(actor('founder'),'ctox.crew.learning.confirm'),true);assert.equal(mayControl(actor('founder'),'ctox.crew.learning.delete'),false);
 assert.equal(mayReadPrivate(actor('user')),false);assert.equal(mayControl(actor('user'),'ctox.crew.assign',{id:'t'}),false);
 const options={...actor('user'),permission:BusinessOsPermissions.CtoxTaskManage,scopeType:'task',scopeId:'t',owned:true,assigned:true};
 assert.equal(canUseBusinessPermission(options),false);
 assert.equal(canUseBusinessPermission({...options,governance:{permission_model:{role_defaults:{user:{owned_task:['ctox.task.manage']}}}}}),false);
 assert.equal(canUseBusinessPermission({...options,governance:{permission_model:{explicit_grants:[{subject_type:'user',subject_id:'operator',permission:'ctox.task.manage',scope_type:'task',scope_id:'t',active:true}]}}}),true);
 assert.equal(mayControl({...actor('user'),governance:{permission_model:{explicit_grants:[{subject_type:'user',subject_id:'operator',permission:'ctox.crew.manage',scope_type:'record',scope_id:'ctox.crew.assign'}]}}},'ctox.crew.assign',{id:'t'}),false);
});
test('leased or reviewing tasks expose no assignment, release, block or retry; cancel needs a command',()=>{
 for(const command of ['ctox.crew.assign','ctox.queue.release','ctox.queue.block','ctox.queue.retry'])assert.equal(taskActionReason(actor('admin'),command,{id:'t',status:'running',route_status:'leased'}),'leasedAction');
 assert.equal(taskActionReason(actor('admin'),'ctox.command.cancel',{id:'t',status:'running'}),'noCommand');
 assert.equal(taskActionReason(actor('admin'),'ctox.crew.assign',{id:'t',status:'blocked'}),'');
});
test('Both locales have identical nonempty keys and interpolation contracts',()=>{
 assert.deepEqual(Object.keys(de).sort(),Object.keys(en).sort());
 for(const key of Object.keys(de)){assert.ok(de[key].trim());assert.ok(en[key].trim());assert.deepEqual(de[key].match(/\{\w+\}/g)?.sort()||[],en[key].match(/\{\w+\}/g)?.sort()||[]);}
 for(const axis of AXES)assert.ok(de['axis_'+axis]);
 const used=new Set();const messages=new Proxy(de,{get(target,key){if(typeof key==='string'){used.add(key);assert.ok(key in target,`missing locale ${key}`);}return target[key];}});
 const s={ctx:actor('admin'),messages,lang:'de',instanceId:'test',view:'home',members:[],tasks:[],events:[],runs:[],memberRuns:[],learnings:[],connected:true,search:'',source:''};
 homeMarkup(s);workplaceMarkup(s);queueMarkup(s);
 const member={id:'m',name:'Milo',shape:'round',color:'#1685ee',state:'home',soul:{sketch:'A',voice:'B'},stats:{}};
 s.members=[member];s.profileId='m';profileMarkup(s);s.tasks=[{id:'t',title:'Task',source:'ctox',status:'pending'}];s.selectedTaskId='t';workplaceMarkup(s);
 assert.ok(used.size>40);
});
test('Retired poster, seed, global refresh, browser task persistence and client redaction are removed',()=>{
 const source=readFileSync(new URL('./index.js',import.meta.url),'utf8');
 assert.doesNotMatch(source,/ctoxSeed|hasSensitiveUiLeak|cleanUiCopy|sessionStorage|setInterval|flowCrewSvg|webStackPanel/);
 assert.match(source,/commandBus\?\.subscribe/);assert.match(source,/\$\?\.subscribe/);
 assert.equal((source.match(/\.innerHTML\s*=/g)||[]).length,1,'only mount replaces host markup');
});

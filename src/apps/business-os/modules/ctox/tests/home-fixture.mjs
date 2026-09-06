/** Isolated browser fixture. No credentials, transport or production database. */
import { mount } from '../index.js';
const lang = new URL(location.href).searchParams.get('lang') === 'en' ? 'en' : 'de';
const role = new URL(location.href).searchParams.get('role') || 'admin';
const t = await (await fetch(`../locales/${lang}.json`)).json();
document.title = t.fixtureTitle;
const watchers = new Set();
const fire = () => watchers.forEach(fn => fn());
let mode = 'idle';
const soul = { gruendlichkeit_vs_tempo:25, vorsicht_vs_mut:30, knapp_vs_ausfuehrlich:60, regeltreu_vs_kreativ:40, nachfragen_vs_annehmen:45, sketch:'Prüft Zusammenhänge und hält Ergebnisse verständlich fest.', voice:'Spricht ruhig und präzise.' };
const members = [['Milo','round','#1685ee'],['Nori','square','#00aa9a'],['Lumi','triangle','#7d7f84'],['Pico','blob','#7c6df2']].map(([name,shape,color],i) => ({id:`fixture-${i}`,name,shape,color,archived:false,state:'home',active_task_id:null,updated_at_ms:100-i,...(role==='user'?{}:{soul:{...soul},specialties:{modules:['ctox'],skills:['Recherche','Validierung'],tags:['Sorgfalt']},stats:{tasks_total:12,succeeded:10,failed:2,review_passed:10,review_rejected:1}})}));
const task = {id:'fixture-task',command_id:'fixture-command',module:'ctox',source_module:'ctox',command_type:'business_os.chat.task',title:'Die nächsten Schritte für das Crew-Zuhause prüfen',prompt:'Prüfe die drei Ansichten und fasse die nächsten sinnvollen Schritte in einem kurzen Plan zusammen.',status:'pending',route_status:'pending',crew_member_id:null,updated_at_ms:Date.now(),priority:'normal',attempt:1};
const data = {ctox_crew_members:members,ctox_queue_tasks:[],business_commands:[],ctox_harness_events:[],ctox_runs:[],ctox_crew_learnings:[{id:'learning-1',member_id:'fixture-0',text:'Erst den aktuellen Zustand erklären, dann die passende Aktion anbieten.',kind:'insight',confirmed_by_owner:false,archived:false,created_at_ms:Date.now(),updated_at_ms:Date.now()}],ctox_harness_status:[{id:'harness',service_running:true,busy:false,paused:false,worker_capacity:1,pending_count:0,leased_count:0,blocked_count:0,active_task_ids:[],active_crew_member_id:null,updated_at_ms:Date.now()}]};
function matches(row, selector) {
  return Object.entries(selector).every(([key,value]) => {
    if (key === '$and') return value.every(q => matches(row,q));
    if (key === '$or') return value.some(q => matches(row,q));
    if (!value || typeof value !== 'object') return row[key] === value;
    return Object.entries(value).every(([op,v]) => {
      if (op === '$eq') return row[key] === v;
      if (op === '$gt') return row[key] > v;
      if (op === '$lt') return row[key] < v;
      if (op === '$in') return v.includes(row[key]);
      if (op === '$regex') return new RegExp(v).test(row[key] || '');
      return false;
    });
  });
}
function collection(name) {
  if (role === 'user' && ['ctox_harness_events','ctox_harness_status','ctox_runs','ctox_crew_learnings'].includes(name)) throw new Error('fixture private access denied');
  return {
    $: { subscribe(fn) { watchers.add(fn); return {unsubscribe() {watchers.delete(fn);}}; } },
    find(query) {
      if (!query.selector || !query.limit) throw new Error('unbounded fixture query');
      return { async exec() {
        if (mode === 'error') throw new Error('fixture: sync_timeout');
        return (data[name] || []).filter(row => matches(row,query.selector)).sort((a,b) => {
          for (const sort of query.sort || []) {
            const [key,dir] = Object.entries(sort)[0];
            if (a[key] !== b[key]) return (a[key]>b[key]?1:-1)*(dir==='desc'?-1:1);
          }
          return 0;
        }).slice(0,query.limit).map(row => ({toJSON:() => structuredClone(row)}));
      }};
    },
  };
}
const ctx = {
  host:document.querySelector('#fixture-host'),locale:lang,session:{user:{id:'fixture-operator',role}},db:{collection},
  sync:{collectionReadiness(){return {ready:mode!=='offline',state:mode==='offline'?'offline-pending':'live'};},subscribeCollectionReadiness(name,fn){watchers.add(fn);return()=>watchers.delete(fn);}},
  commandBus:{
    subscribe(){return {unsubscribe(){}};},
    async dispatch(command){
      const p=command.payload;const type=command.command_type;
      document.querySelector('#fixture-controls').dataset.lastCommand=type;
      if(type==='ctox.command.cancel')transition('cancelled');
      if(type==='ctox.queue.block'){task.status='blocked';task.route_status='blocked';task.hold_reason=p.reason;}
      if(type==='ctox.queue.release'||type==='ctox.queue.retry'){task.status='pending';task.route_status='pending';task.hold_reason=null;}
      if(type==='ctox.queue.pause')data.ctox_harness_status[0].paused=p.paused;
      if(type==='ctox.queue.capacity')data.ctox_harness_status[0].worker_capacity=p.workers;
      if(type==='ctox.task.update')task.priority=p.priority;
      if(type==='ctox.crew.assign')task.crew_assigned_member_id=p.member_id;
      if(type==='ctox.crew.member.update')Object.assign(members.find(m=>m.id===p.member_id),Object.fromEntries(Object.entries(p).filter(([k])=>k!=='member_id')));
      if(type==='ctox.crew.learning.confirm')data.ctox_crew_learnings.find(l=>l.id===p.learning_id).confirmed_by_owner=true;
      if(type==='ctox.crew.learning.update')data.ctox_crew_learnings.find(l=>l.id===p.learning_id).text=p.text;
      if(type==='ctox.crew.learning.delete')data.ctox_crew_learnings=data.ctox_crew_learnings.filter(l=>l.id!==p.learning_id);
      fire();return {status:'completed'};
    },
  },
};
function transition(next) {
  mode=next;const active=['working','review'].includes(next);const terminal=['done','failed','cancelled'].includes(next);
  Object.assign(task,{status:active?(next==='review'?'review':'running'):terminal?(next==='done'?'completed':next):'pending',route_status:active?'leased':terminal?'done':'pending',crew_member_id:active||terminal?'fixture-0':null,updated_at_ms:Date.now()});
  task.execution_progress=active||terminal?{phase:active?'working':'terminal',steps:[{position:1,label:'Bestehende Verträge lesen',status:'completed'},{position:2,label:'Oberfläche im Browser prüfen',status:active?'in_progress':'completed'},{position:3,label:'Ergebnisse festhalten',status:active?'pending':'completed'}],review:{status:next==='review'?'validating':'pending'}}:null;
  Object.assign(members[0],{state:active?'on_duty':next==='failed'?'resting_after_failure':'home',active_task_id:active?task.id:null});
  Object.assign(data.ctox_harness_status[0],{busy:active,active_crew_member_id:active?'fixture-0':null,active_task_ids:active?[task.id]:[],pending_count:next==='queued'?1:0,leased_count:active?1:0,updated_at_ms:Date.now()});
  data.ctox_queue_tasks=next==='idle'?[]:[task];
  data.business_commands=next==='idle'?[]:[{id:'fixture-command',command_id:'fixture-command',execution_mode:'queue',execution_task_id:task.id,execution_phase:active?(next==='review'?'validating':'running'):terminal?'terminal':'queued',terminal_status:terminal?task.status:'none',status:terminal?task.status:'accepted',module:'ctox',command_type:task.command_type,updated_at_ms:Date.now()}];
  data.ctox_harness_events=active?[{id:'event-1',task_id:task.id,kind:'thinking',title:'Die Prüfschritte werden vorbereitet.',created_at_ms:Date.now()-1000,updated_at_ms:Date.now()}]:[];
  data.ctox_runs=terminal?[{id:'attempt-1',task_id:task.id,crew_member_id:'fixture-0',status:task.status,finished_at_ms:Date.now(),updated_at_ms:Date.now(),metrics:{model:'Fixture model',input_tokens:3200,output_tokens:850,reasoning_tokens:450,cost_usd:.024,elapsed_ms:41000},review:{disposition:'passed'},retrospective:'Die Zustände sind klar voneinander getrennt.'}]:[];
  fire();
}
for (const next of ['idle','queued','working','review','done','failed','offline','error']) {
  const button=document.createElement('button');button.type='button';button.textContent=t['fixture_'+next];button.addEventListener('click',()=>transition(next));document.querySelector('#fixture-controls').append(button);
}
await mount(ctx);

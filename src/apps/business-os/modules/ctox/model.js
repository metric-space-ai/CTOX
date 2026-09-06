import { businessActorFromSession, canUseBusinessPermission, BusinessOsPermissions } from '../../shared/permissions.js';
export const PAGE_SIZE = 40;
export const TERMINAL = ['completed','done','succeeded','failed','cancelled','canceled','sent','approved'];
export const ACTIVE = ['pending','queued','accepted','leased','running','working','blocked','review','validating','waiting','deferred'];
const numeric = value => value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value) : null;
const clamp = value => value === null ? null : Math.max(0,Math.min(100,value));

export function normalizeExecutionProgress(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const turns=raw.activity_turns || raw.activityTurns || {};
  const steps=(Array.isArray(raw.steps)?raw.steps:[]).map((step,index)=>({
    position:numeric(step.position) ?? index+1,label:String(step.label || ''),status:String(step.status || ''),
    activityTurns:numeric(step.activity_turns ?? step.activityTurns),
  }));
  const reviewStatus=String(raw.review?.status || raw.review_status || raw.reviewStatus || '');
  if (!steps.length && numeric(raw.percent)===null && !reviewStatus && !raw.phase && !Object.keys(turns).length) return null;
  return {phase:String(raw.phase || ''),steps,reviewStatus,percent:clamp(numeric(raw.percent)),
    currentStep:numeric(raw.current_step ?? raw.currentStep),totalSteps:numeric(raw.total_steps ?? raw.totalSteps),
    completedSteps:numeric(raw.completed_steps ?? raw.completedSteps),thinkingTurns:numeric(turns.thinking),toolTurns:numeric(turns.tools),
    totalTurns:numeric(turns.total),updatedAtMs:numeric(raw.updated_at_ms ?? raw.updatedAtMs)};
}
export function queueTaskFromProjection(doc) {
  return {...doc,id:doc.id || doc.task_id,taskId:doc.task_id || doc.id,commandId:doc.command_id || '',
    source:doc.source_module || doc.module || '',routeStatus:doc.route_status || '',
    executionProgress:normalizeExecutionProgress(doc.execution_progress),updatedAtMs:numeric(doc.updated_at_ms)};
}
export function commandTaskFromProjection(doc, runtimeByTaskId=new Map(), runtimeByCommandId=new Map()) {
  const commandId=String(doc.command_id || doc.id || '');
  if (!commandId || doc.execution_mode==='control') return null;
  const taskId=doc.execution_task_id || (doc.contract_version!==2 ? doc.task_id : '') || '';
  const queue=runtimeByTaskId.get(taskId) || runtimeByCommandId.get(commandId);
  if (!queue && !taskId) return null;
  const phase=String(doc.execution_phase || '');
  const status=phase==='terminal' ? (doc.terminal_status || doc.status) : phase==='running' ? 'running' : phase==='validating' ? 'review' : (queue?.status || doc.status);
  return {...(queue || {}),id:queue?.id || taskId,taskId:queue?.taskId || taskId,
    commandId,command_id:commandId,command_type:doc.command_type || queue?.command_type,
    title:doc.payload?.title || queue?.title || '',prompt:doc.payload?.instruction || queue?.prompt || '',
    source:doc.module || queue?.source || '',module:doc.module || queue?.module,
    status,routeStatus:phase ? (phase==='terminal'?status:phase) : queue?.routeStatus || '',
    executionPhase:phase,execution_phase:phase,terminal_status:doc.terminal_status,
    executionProgress:normalizeExecutionProgress(doc.execution_progress) || queue?.executionProgress || null,
    updated_at_ms:numeric(doc.updated_at_ms) ?? queue?.updated_at_ms,
    updatedAtMs:numeric(doc.updated_at_ms) ?? queue?.updatedAtMs};
}
/** Queue telemetry stays attached when lifecycle-v2 advances ahead of it. */
export function mergeBundleWithCommands(bundle={},commands=[],queueTasks=[]) {
  const tasks=queueTasks.map(queueTaskFromProjection).filter(task=>task.id);
  const byId=new Map(tasks.map(task=>[task.id,task]));
  const byCommand=new Map(tasks.filter(task=>task.commandId).map(task=>[task.commandId,task]));
  for (const command of commands) {
    const task=commandTaskFromProjection(command,byId,byCommand);
    if(task) byId.set(task.id,task);
  }
  return {...bundle,queue:[...byId.values()].sort((a,b)=>(b.updatedAtMs || 0)-(a.updatedAtMs || 0) || String(b.id).localeCompare(String(a.id)))};
}
export function taskGroup(task) {
  if (['failed','error'].includes(task.status)) return 'failed';
  if (TERMINAL.includes(task.status)) return 'done';
  if (task.status==='blocked' || task.route_status==='blocked') return 'blocked';
  if ([task.status,task.routeStatus,task.executionPhase,task.executionProgress?.reviewStatus].some(x=>['review','reviewing','validating','awaiting_review'].includes(x))) return 'review';
  if ([task.status,task.routeStatus,task.executionPhase].some(x=>['running','leased','working','drafting'].includes(x))) return 'running';
  return 'waiting';
}
export function mayReadPrivate(ctx) { return ['admin','chef','founder'].includes(businessActorFromSession(ctx.session,ctx.governance).role); }
export function mayControl(ctx,command,task) {
  const crew=command.startsWith('ctox.crew.');
  return canUseBusinessPermission({session:ctx.session,governance:ctx.governance,
    permission:crew?BusinessOsPermissions.CrewManage:BusinessOsPermissions.CtoxTaskManage,
    scopeType:crew?'record':task?'task':'workspace',scopeId:crew?command:task?.id || '',owned:false,assigned:false});
}
export function isLeased(task,now=Date.now()) {
  return task?.route_status==='leased' || task?.routeStatus==='leased' || (task?.lease_owner && (!task.lease_expires_at || Date.parse(task.lease_expires_at)>now));
}
export function taskActionReason(ctx,command,task) {
  if (!mayControl(ctx,command,task)) return 'permissionDenied';
  if (!task) return 'noTask';
  if (command==='ctox.command.cancel') return !task.commandId?'noCommand':TERMINAL.includes(task.status)?'alreadyTerminal':'';
  if (command==='ctox.task.update') return TERMINAL.includes(task.status)?'alreadyTerminal':'';
  if (isLeased(task) || ['running','review'].includes(taskGroup(task))) return 'leasedAction';
  if(command==='ctox.crew.assign') return ['pending','queued','blocked'].includes(task.status)?'':'assignmentUnavailable';
  if(command==='ctox.queue.release' && TERMINAL.includes(task.status)) return 'alreadyTerminal';
  if(command==='ctox.queue.block' && TERMINAL.includes(task.status)) return 'alreadyTerminal';
  return '';
}

function page(selector,field,limit,cursor) {
  const clauses=[selector];
  if(cursor) clauses.push({$or:[{[field]:{$lt:cursor[field]}},{$and:[{[field]:{$eq:cursor[field]}},{id:{$lt:cursor.id}}]}]});
  return {selector:clauses.length===1?selector:{$and:clauses},sort:[{[field]:'desc'},{id:'desc'}],limit};
}
export const queries={
  members:cursor=>page({id:{$gt:''}},'updated_at_ms',40,cursor),
  status:()=>({selector:{id:{$eq:'harness'}},limit:1}),
  tasks:({source='',search='',terminal=false,cursor}={})=>{
    const clauses=[{status:{$in:terminal?TERMINAL:ACTIVE}}];
    if(source) clauses.push({module:{$eq:source}});
    if(search.trim()) clauses.push({title:{$regex:search.trim().replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}});
    return page({$and:clauses},'updated_at_ms',PAGE_SIZE,cursor);
  },
  task:id=>({selector:{id:{$eq:id}},limit:1}),
  commands:ids=>({selector:{command_id:{$in:ids}},limit:Math.max(1,Math.min(200,ids.length))}),
  activeCommands:()=>page({execution_mode:{$eq:'queue'},execution_phase:{$in:['accepted','queued','running','validating']}},'updated_at_ms',40),
  events:taskId=>page({task_id:{$eq:taskId}},'created_at_ms',200),
  runs:({taskId,memberId,cursor})=>page(taskId?{task_id:{$eq:taskId}}:{crew_member_id:{$eq:memberId}},'finished_at_ms',20,cursor),
  learnings:(memberId,cursor)=>page({member_id:{$eq:memberId}},'created_at_ms',40,cursor),
};
/** All reads use shell handles. Refuse unbounded accidental future callers. */
export async function readCollection(ctx,name,query) {
  if(!query?.selector || !Object.keys(query.selector).length || !Number.isInteger(query.limit) || query.limit<1 || query.limit>200) throw new Error('invalid_bounded_query');
  const collection=ctx.db?.collection?.(name);
  if(!collection?.find) throw new Error(`collection_unavailable: ${name}`);
  const rows=await collection.find(query).exec();
  return rows.map(row=>row.toJSON?.() || row);
}

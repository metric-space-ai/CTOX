import {crewCreatureHtml,CREATURE_STATES} from '../../../shared/crew-creature.js';
const lang=new URL(location.href).searchParams.get('lang')==='en'?'en':'de';
const t=await (await fetch(`../locales/${lang}.json`)).json();
document.documentElement.lang=lang;
document.title=t.creatureGallery;
document.querySelector('h1').textContent=t.creatureGallery;
// Explicit fixture identities: this gallery never reads or writes Business OS.
const members=[['Milo','round','#1685ee'],['Nori','square','#00aa9a'],['Lumi','triangle','#7d7f84'],['Pico','blob','#7c6df2']];
for(const [name,shape,color] of members) for(const state of CREATURE_STATES){
 const el=document.createElement('figure');
 el.innerHTML=crewCreatureHtml({id:`fixture-${name}`,name,shape,color},{state,label:`${name}: ${t['creature_'+state]}`});
 const caption=document.createElement('figcaption');caption.textContent=`${name} · ${t['creature_'+state]}`;el.append(caption);document.querySelector('main').append(el);
}

import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { crc32 } from 'node:zlib';
import { chromium } from '../node_modules/playwright/index.mjs';

// Serve the repository root on port 8766 before running this harness. It uses
// isolated mock collections/commands, real editor frames and the native CLI;
// this is not tenant WebRTC/permission or deployment verification.

const root = fileURLToPath(new URL('../../../../', import.meta.url));
const engineBin = process.argv.find(value => value.startsWith('--engine-bin='))?.slice('--engine-bin='.length)
  || path.join(root, 'runtime/build/cargo-target/debug/ctox-office-engine');
const output = path.join(root, 'output/playwright/office-integration');
await mkdir(output, { recursive: true });
const temporary = await mkdtemp(path.join(output, 'native-'));
const run = promisify(execFile);

// Isolated equivalent of the server's empty CSV -> OOXML conversion. The real
// native DOCY/XLSY converter still prepares every newly created editor payload.
function emptyWorkbook() {
  const parts={
    '[Content_Types].xml':'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
    '_rels/.rels':'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
    'xl/workbook.xml':'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Tabelle1" sheetId="1" r:id="rId1"/></sheets></workbook>',
    'xl/_rels/workbook.xml.rels':'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
    'xl/worksheets/sheet1.xml':'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1"/><sheetData/></worksheet>',
    'xl/styles.xml':'<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
  };
  parts['[Content_Types].xml']=parts['[Content_Types].xml'].replace('</Types>','<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>');
  parts['xl/_rels/workbook.xml.rels']=parts['xl/_rels/workbook.xml.rels'].replace('</Relationships>','<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>');
  const local=[],central=[]; let offset=0;
  for(const [name,xml] of Object.entries(parts)) {
    const filename=Buffer.from(name),data=Buffer.from(xml),checksum=crc32(data);
    const header=Buffer.alloc(30);header.writeUInt32LE(0x04034b50);header.writeUInt16LE(20,4);header.writeUInt32LE(checksum,14);header.writeUInt32LE(data.length,18);header.writeUInt32LE(data.length,22);header.writeUInt16LE(filename.length,26);
    const directory=Buffer.alloc(46);directory.writeUInt32LE(0x02014b50);directory.writeUInt16LE(20,4);directory.writeUInt16LE(20,6);directory.writeUInt32LE(checksum,16);directory.writeUInt32LE(data.length,20);directory.writeUInt32LE(data.length,24);directory.writeUInt16LE(filename.length,28);directory.writeUInt32LE(offset,42);
    local.push(header,filename,data);central.push(directory,filename);offset+=header.length+filename.length+data.length;
  }
  const directory=Buffer.concat(central),end=Buffer.alloc(22);end.writeUInt32LE(0x06054b50);end.writeUInt16LE(Object.keys(parts).length,8);end.writeUInt16LE(Object.keys(parts).length,10);end.writeUInt32LE(directory.length,12);end.writeUInt32LE(offset,16);
  return Buffer.concat([...local,directory,end]);
}
const browser = await chromium.launch({ headless: true });
const failures = [];
let activePage;
let activeErrors;
try {
  for (const kind of ['spreadsheet', 'document']) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
    await context.exposeBinding('officeLabPrepare', async (_, requestedKind, encoded) => {
      assert.equal(requestedKind, kind);
      let source = Buffer.from(encoded, 'base64');
      if(kind==='spreadsheet' && source.subarray(0,2).toString()!=='PK') {
        assert.match(source.toString(), /^[,\r\n]*$/, 'New spreadsheet must contain no sample data');
        source=emptyWorkbook();
      }
      const input = path.join(temporary, `${kind}.input`);
      const result = path.join(temporary, `${kind}.bin`);
      await writeFile(input, source);
      await run(engineBin, ['prepare-editor', kind, input, result]);
      return (await readFile(result)).toString('base64');
    });
    const page = await context.newPage();
    page.setDefaultTimeout(15000);
    activePage = page;
    await page.addInitScript(() => window.addEventListener('message', event => {
      let message=event.data;
      if(typeof message==='string') { try { message=JSON.parse(message); } catch { return; } }
      if(message?.event==='onDocumentReady') window.__officeLabReady=true;
    }));
    const errors = [];
    activeErrors = errors;
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', event => { if(event.type()==='error') errors.push(event.text()); });
    page.on('response', response => { if(response.status()>=400) console.log(JSON.stringify({kind,failedAsset:response.url(),status:response.status()})); });
    await page.goto(`http://127.0.0.1:8766/src/apps/business-os/office-engine/oracle/shell-v2-office.html?kind=${kind}`);
    const prefix = kind === 'document' ? 'documents' : 'spreadsheets';
    await page.locator(`.${prefix}-card-main`).first().click({timeout:20000});
    const outer = page.frameLocator(`iframe[data-ctox-office-kind="${kind}"]`);
    const editor = outer.frameLocator('iframe.ctox-office-fork-frame');
    await editor.locator('#viewport').waitFor({state:'visible',timeout:60000});
    await editor.getByRole('tab', {name:'Startseite',exact:true}).waitFor({state:'visible',timeout:30000});
    const runtimeFrame=page.frames().find(frame=>frame.parentFrame()===page.mainFrame());
    await runtimeFrame.waitForFunction(()=>window.__officeLabReady===true,null,{timeout:30000});
    const geometry = await page.locator('[data-shell-columns="2"]').evaluate(root => {
      const library=root.querySelector('.ctox-office-library'), editor=root.querySelector('.documents-workbench,.spreadsheets-editor');
      return {root:root.getBoundingClientRect().toJSON(),library:library.getBoundingClientRect().toJSON(),editor:editor.getBoundingClientRect().toJSON(),display:getComputedStyle(root).display,columns:getComputedStyle(root).gridTemplateColumns};
    });
    console.log(JSON.stringify({kind,geometry,errors}));
    assert.ok(geometry.editor.x > geometry.library.x + geometry.library.width - 1, 'Editor must be beside the library');
    assert.ok(geometry.editor.width > 500, 'Editor gets the available work area');
    await page.screenshot({path:path.join(output,`${kind}-light.png`)});
    assert.equal(await page.locator('.ctox-office-library').evaluate(library=>getComputedStyle(library.querySelector('.ctox-pane-header')).backgroundColor===getComputedStyle(library).backgroundColor),true,'Library header must use the same light-theme surface');
    const search=page.locator('.ctox-office-library [data-pg-search]');
    await search.fill('Arbeitsdatei 2');
    assert.equal(await page.locator(`.${prefix}-card-main`).count(),10);
    assert.equal(await search.evaluate(node=>node===document.activeElement),true);
    await search.fill('');
    assert.equal(await page.locator(`.${prefix}-card-main`).count(),30);
    await page.locator('[data-pg-tray-toggle]').click();
    assert.equal(await page.locator('[data-pg-tray]').isVisible(),true);
    await page.locator('[data-pg-tray-toggle]').click();
    await page.locator('[data-pg-view-cycle]').click();
    assert.equal(await page.locator('.ctox-office-library').getAttribute('data-office-view'),'cards');
    await page.locator('[data-pg-view-cycle]').click();
    await page.evaluate(()=>document.documentElement.dataset.theme='dark');
    await editor.locator('html[data-ctox-theme="dark"]').waitFor({state:'attached',timeout:5000});
    await page.locator('.ctox-office-library').evaluate(async library=>{await Promise.all(library.getAnimations({subtree:true}).map(animation=>animation.finished.catch(()=>{})));});
    await page.screenshot({path:path.join(output,`${kind}-dark.png`)});
    assert.equal(await editor.locator('#toolbar button svg').first().evaluate(svg=>getComputedStyle(svg).color===getComputedStyle(document.body).color),true,'Monochrome SVG icons must follow Shell text contrast');
    console.log(`${kind}: theme checked; checking custom palette`);
    await page.evaluate(()=>document.querySelector('#app').style.setProperty('--accent','#a855f7'));
    const innerFrame=page.frames().find(frame=>frame.parentFrame()===runtimeFrame);
    await innerFrame.waitForFunction(()=>getComputedStyle(document.documentElement).getPropertyValue('--ctox-shell-accent').trim()==='#a855f7');
    assert.equal(await editor.locator('body').evaluate((body, kind) => {
      const style=getComputedStyle(body),surface=style.getPropertyValue('--ctox-fork-surface-2').trim();
      return style.getPropertyValue('--background-toolbar').trim()===surface && style.getPropertyValue(`--toolbar-header-${kind}`).trim()===surface;
    },kind),true,'Native ribbon colors must inherit the Shell surface, not override it with a product theme');
    for(const width of [960,640,390]) {
      console.log(`${kind}: checking width ${width}`);
      await page.setViewportSize({width,height:800});
      const toggle=page.locator(kind==='document'?'[data-documents-library-toggle]':'[data-spreadsheets-library-toggle]');
      if(width<=768) {
        await toggle.waitFor({state:'visible'});
        await toggle.click();
        assert.equal(await search.isVisible(),true);
        await page.locator(kind==='document'?'[data-documents-drawer-backdrop]':'[data-spreadsheets-library-backdrop]').click({position:{x:width-20,y:500}});
      }
      const bounds=await page.locator(`.${prefix}-module`).boundingBox();
      assert.ok(bounds.width<=width,'Office root must fit the window');
      await page.screenshot({path:path.join(output,`${kind}-${width}.png`)});
    }
    await page.setViewportSize({width:1440,height:960});
    console.log(`${kind}: creating blank file`);
    const previousFrame=await page.locator(`iframe[data-ctox-office-kind="${kind}"]`).elementHandle();
    await page.locator(kind==='document'?'[data-documents-new-markdown]':'[data-spreadsheets-new]').click();
    await page.waitForFunction(()=>window.officeLab.records.length===31);
    await page.waitForFunction(()=>window.officeLab.commands.some(command=>command.type.endsWith('.prepare')));
    await previousFrame.waitForElementState('hidden');
    await editor.locator('#viewport').waitFor({state:'visible',timeout:30000});
    const blankRuntime=page.frames().find(frame=>frame.parentFrame()===page.mainFrame());
    await blankRuntime.waitForFunction(()=>window.__officeLabReady===true,null,{timeout:30000});
    assert.equal(await page.locator('#lab-drawer').count(),0,'Blank creation must not require a prompt dialog');
    const replacement=`CTOX_OFFICE_${kind.toUpperCase()}_SAVED`;
    if(kind==='spreadsheet') {
      await editor.locator('#ce-cell-name').fill('A1'); await editor.locator('#ce-cell-name').press('Enter');
      assert.equal(await editor.locator('#ce-cell-content').inputValue(),'');
      await editor.locator('#ce-cell-content').fill(replacement);await editor.locator('#ce-cell-content').press('Enter');
    } else {
      const canvas=editor.locator('#id_viewer_overlay');const box=await canvas.boundingBox();
      await canvas.click({position:{x:box.width*0.35,y:120}});await page.keyboard.type(replacement);
    }
    await editor.getByRole('button',{name:'Speichern (⌘+S)',exact:true}).click();
    await page.waitForFunction(()=>window.officeLab.commands.some(command=>command.type.endsWith('.commit')),null,{timeout:15000});
    const saved=await page.evaluate(()=>{
      const command=window.officeLab.commands.findLast(command=>command.type.endsWith('.commit'));
      return window.officeLab.chunks.filter(row=>row.blob_id===command.payload.editor_blob_id).sort((a,b)=>a.idx-b.idx).map(row=>row.data);
    });
    const savedFile=path.join(temporary,`${kind}-saved.bin`);
    await writeFile(savedFile,Buffer.concat(saved.map(value=>Buffer.from(value,'base64'))));
    const inspection=await run(engineBin,['inspect-editor',kind,savedFile]);
    assert.equal(JSON.parse(inspection.stdout).kind,kind);
    if(kind==='document') {
      const exported=path.join(temporary,'document-saved.docx');
      await run(engineBin,['export',kind,savedFile,path.join(temporary,'document.input'),exported]);
      const xml=await run('unzip',['-p',exported,'word/document.xml']);
      assert.ok(xml.stdout.replace(/<[^>]+>/g,'').includes(replacement),'Native DOCX export must preserve the typed text');
    } else {
      assert.ok(inspection.stdout.includes(replacement),'Native inspection must find the cell entered through the UI');
      const exported=path.join(temporary,'spreadsheet-saved.xlsx');
      await run(engineBin,['export',kind,savedFile,path.join(temporary,'spreadsheet.input'),exported]);
      const reopenedBinary=path.join(temporary,'spreadsheet-export-reopened.bin');
      await run(engineBin,['prepare-editor',kind,exported,reopenedBinary]);
      const reopenedInspection=await run(engineBin,['inspect-editor',kind,reopenedBinary]);
      assert.ok(reopenedInspection.stdout.includes(replacement),'Native XLSX export and re-import must preserve the typed cell');
    }
    await page.screenshot({path:path.join(output,`${kind}-blank-edited.png`)});
    const savedIdentity=await page.evaluate(()=>{
      const record=window.officeLab.records.find(row=>row.title.startsWith('Neu'));
      const version=window.officeLab.versions.find(row=>row.id===record.current_version_id);
      return {id:record.id,version:version.id,editorBlob:version.editor_blob_id};
    });
    await page.reload();
    await page.locator(`.${prefix}-card-main`).first().click();
    await editor.locator('#viewport').waitFor({state:'visible',timeout:30000});
    const reopenedRuntime=page.frames().find(frame=>frame.parentFrame()===page.mainFrame());
    await reopenedRuntime.waitForFunction(()=>window.__officeLabReady===true,null,{timeout:30000});
    assert.equal(await page.evaluate(({id,version,editorBlob})=>{
      const record=window.officeLab.records.find(row=>row.id===id);
      return record.current_version_id===version && window.officeLab.versions.find(row=>row.id===version).editor_blob_id===editorBlob;
    },savedIdentity),true,'Reopening must retain the saved version, not the original blank');
    if(kind==='spreadsheet') {
      await editor.locator('#ce-cell-name').fill('A1');await editor.locator('#ce-cell-name').press('Enter');
      assert.equal(await editor.locator('#ce-cell-content').inputValue(),replacement);
      const chrome = await editor.locator('#ce-cell-content').evaluate(input => {
        const box = node => node ? {id:node.id,classes:node.className,rect:node.getBoundingClientRect().toJSON(),overflow:getComputedStyle(node).overflow,color:getComputedStyle(node).color,fontSize:getComputedStyle(node).fontSize,lineHeight:getComputedStyle(node).lineHeight} : null;
        const textBox = label => {
          const text=[...label.childNodes].find(node=>node.nodeType===Node.TEXT_NODE && node.textContent.trim());
          if(!text) return null;
          const range=document.createRange();range.selectNodeContents(text);
          return range.getBoundingClientRect().toJSON();
        };
        return {input:box(input),parent:box(input.parentElement),grandparent:box(input.parentElement.parentElement),toolbar:box(document.querySelector('#toolbar')),panels:[...document.querySelectorAll('#toolbar .panel,#toolbar .box-controls,.toolbar-fullview-panel')].map(box).filter(node=>node.rect.width && node.rect.height),statusbar:box(document.querySelector('#statusbar')),statusTabs:[...document.querySelectorAll('#statusbar li.list-item')].map(node=>({text:node.textContent,item:box(node),label:box(node.querySelector('span')),textRect:textBox(node.querySelector('span'))}))};
      });
      console.log(JSON.stringify({kind,chrome}));
      assert.ok(chrome.input.rect.top >= chrome.toolbar.rect.bottom - 1, 'Formula bar must not be covered by the ribbon');
      assert.ok(chrome.panels.every(panel=>panel.rect.bottom <= chrome.input.rect.top + 1), 'Visible ribbon panels must not overlap the formula input');
      assert.ok(chrome.statusTabs.length > 0, 'Workbook must expose a named sheet tab');
      // The label has an intentional offscreen ::after width-measuring copy.
      // Assert the real text glyphs, not that hidden measurement box.
      assert.ok(chrome.statusTabs.every(tab=>tab.textRect?.width > 0 && tab.textRect.top >= chrome.statusbar.rect.top && tab.textRect.bottom <= chrome.statusbar.rect.bottom), 'Sheet label text must fit inside the status bar, not below the viewport');
    }
    await page.screenshot({path:path.join(output,`${kind}-reopened.png`)});
    console.log(JSON.stringify({kind,flows:'open, search, filter, view, theme, custom accent, responsive, create blank, keyboard edit, save, native export/inspection, reopen'}));
    assert.deepEqual(errors, []);
    await context.close();
  }
} catch(error) {
  console.log(JSON.stringify({errors:activeErrors,frames:activePage?.frames().map(frame=>frame.url())}));
  for(const frame of activePage?.frames() || []) console.log(JSON.stringify(await frame.evaluate(()=>({url:location.href,theme:window.uitheme?.id,bodyClass:document.body.className,accent:getComputedStyle(document.querySelector('[data-ctox-office-kind]') || document.documentElement).getPropertyValue('--accent'),shellAccent:getComputedStyle(document.documentElement).getPropertyValue('--ctox-shell-accent'),styles:document.documentElement.getAttribute('style')}))));
  if(activePage) { console.log((await activePage.locator('body').innerText()).slice(-6000)); await activePage.screenshot({path:path.join(output,'failure.png')}); }
  failures.push(error.stack);
}
finally { await browser.close(); }
if(failures.length) throw new Error(failures.join('\n'));

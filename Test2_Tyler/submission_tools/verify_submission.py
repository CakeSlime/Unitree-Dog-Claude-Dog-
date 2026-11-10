"""Read-only evidence/preservation checks; write only final submission audit JSON."""
from pathlib import Path
import hashlib,json,re,subprocess,zipfile
from xml.etree import ElementTree
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

result={}
(ROOT/'report/staging_batches').mkdir(exist_ok=True)
before=json.loads((ROOT/'report/shared_Test2_before.sha256.json').read_text())
current={p.relative_to(REPO).as_posix() for p in (REPO/'Test2').rglob('*') if p.is_file()}
changed=[n for n,h in before.items() if not (REPO/n).is_file() or digest(REPO/n)!=h]
result['shared_Test2_preservation']={'baseline_files':len(before),'current_files':len(current),
 'changed_or_missing':changed,'added':sorted(current-set(before)),'unchanged':not changed and current==set(before)}
recordings=[];total=0
for manifest in sorted((ROOT/'data').glob('*/*/sha256.json')):
 hashes=json.loads(manifest.read_text());bad=[]
 for n,h in hashes.items():
  p=manifest.parent/n
  if not p.is_file() or digest(p)!=h:bad.append(n)
 total+=len(hashes)
 recordings.append({'trial':manifest.parent.relative_to(ROOT).as_posix(),'files':len(hashes),'mismatches':bad})
result['recording_manifests']={'trials':recordings,'files_checked':total,'all_match':all(not r['mismatches'] for r in recordings)}
# All raw stream files must appear in the delivered ZIP parts, including interrupted trials.
m=json.loads((ROOT/'recording_archives/manifest.json').read_text());archives={}
for trial in m['trials']:
 archives[trial['trial']]=trial['files']
raw={}
for phase in ('sample','phase2b','phase2c'):
 for trial in (ROOT/'data'/phase).iterdir():
  if trial.is_dir():
   count=sum(1 for folder in ('rgb','depth','legacy_depth','lidar') for p in (trial/folder).glob('*') if p.is_file())
   if count:raw[trial.relative_to(ROOT).as_posix()]=count
result['archive_coverage']={'all_raw_streams_included':raw==archives,'raw_files':sum(raw.values()),
 'parts':sum(len(t['parts']) for t in m['trials']),'bytes':sum(p['bytes'] for t in m['trials'] for p in t['parts'])}
links=[]
for name in ('README.md','report/ENGR6010_Second_Test_Final_Report.md','report/Final_Submission_Audit.md','report/Phase2C_Implementation_Report.md','phase2c/README.md'):
 p=ROOT/name
 for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in link or link.startswith('#'):continue
  link=link.split('#')[0]
  if link and not (p.parent/link).exists():links.append({'document':name,'missing':link})
result['broken_current_document_links']=links
suite=ElementTree.parse(ROOT/'report/submission_tests.xml').getroot()[0]
result['pytest']={k:suite.attrib[k] for k in ('tests','failures','errors','skipped')}
with zipfile.ZipFile(ROOT/'report/ENGR6010_Second_Test_Final_Report.docx') as z:
 assert z.testzip() is None
 for n in ('word/document.xml','word/styles.xml','word/_rels/document.xml.rels','[Content_Types].xml'):
  ElementTree.fromstring(z.read(n))
result['docx_xml_valid']=True
result['pdf_signature_valid']=(ROOT/'report/ENGR6010_Second_Test_Final_Report.pdf').read_bytes().startswith(b'%PDF-')
result['git_branch']=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
# Enumerate explicit submission paths, respecting ignores; no other project deliverables.
paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--','.gitignore','Test2_Tyler'],cwd=REPO,text=True).splitlines()
# Reserve paths for generated audit and explicit batch lists.
paths=sorted(set(paths+['Test2_Tyler/report/submission_integrity.json','Test2_Tyler/report/staging_paths.txt']))
archive_paths=[n for n in paths if '.zip.part' in n]
core=[n for n in paths if '.zip.part' not in n and '/staging_batches/' not in n]
batches=[];batch=[];size=0
for n in archive_paths:
 amount=(REPO/n).stat().st_size
 if batch and size+amount>600*1024*1024:
  batches.append(batch);batch=[];size=0
 batch.append(n);size+=amount
if batch:batches.append(batch)
batch_dir=ROOT/'report/staging_batches'
batch_names=['Test2_Tyler/report/staging_batches/core.txt']+[f'Test2_Tyler/report/staging_batches/data_{i:03d}.txt' for i in range(1,len(batches)+1)]
core=sorted(set(core+batch_names))
(batch_dir/'core.txt').write_text('\n'.join(core)+'\n')
for i,batch in enumerate(batches,1):
 (batch_dir/f'data_{i:03d}.txt').write_text('\n'.join(batch)+'\n')
paths=sorted(set(core+archive_paths))
large=[{'path':n,'bytes':(REPO/n).stat().st_size} for n in paths if (REPO/n).exists() and (REPO/n).stat().st_size>=50*1024*1024]
secrets=[n for n in paths if any(part in {'.cache','.venv','.aws','.codex','.agents'} for part in Path(n).parts) or Path(n).name.startswith('.env') or Path(n).suffix in {'.pem','.key'}]
sensitive=[]
pattern=re.compile(r'(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
for n in paths:
 p=REPO/n
 if p.exists() and p.suffix in {'.py','.md','.txt','.json','.yaml','.yml','.csv','.log'} and p.stat().st_size<2*1024*1024:
  if pattern.search(p.read_text(errors='replace')):sensitive.append(n)
result['sensitive_content_matches']=sensitive
result['publication_batches']={'core_paths':len(core),'data_batches':[{'name':f'data_{i:03d}.txt','parts':len(batch),'bytes':sum((REPO/n).stat().st_size for n in batch)} for i,batch in enumerate(batches,1)]}
result['staging_audit']={'paths':len(paths),'bytes':sum((REPO/n).stat().st_size for n in paths if (REPO/n).exists()),
 'files_at_least_50_MiB':large,'credential_or_cache_paths':secrets}
result['passed']=bool(result['shared_Test2_preservation']['unchanged'] and result['recording_manifests']['all_match'] and raw==archives and not links and not large and not secrets and not sensitive and suite.attrib['failures']=='0' and suite.attrib['errors']=='0')
(ROOT/'report/submission_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
# Include generated audit and allowlist themselves even on first run.
paths=sorted(set(paths+['Test2_Tyler/report/submission_integrity.json','Test2_Tyler/report/staging_paths.txt']))
(ROOT/'report/staging_paths.txt').write_text('\n'.join(paths)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='recording_manifests'},indent=2))
if not result['passed']:raise SystemExit(2)

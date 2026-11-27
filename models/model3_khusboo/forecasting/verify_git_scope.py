"""Verify Model 1/2 preservation and byte-exact staging of model artifacts."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
BASE='117a35f5d3098e5dc9278c9002896c406a41f315'
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
def main():
    rows=git('diff','--cached','--name-status').splitlines()
    if not rows:rows=git('diff',BASE,'HEAD','--name-status').splitlines()
    assert rows
    for row in rows:
        status,name=row.split('\t',1)
        assert name.startswith('models/model3_khusboo/'),name
        assert status=='A' or (status=='M' and name=='models/model3_khusboo/README.md'),row
    index={r.split('\t',1)[1]:r.split()[1] for r in git('ls-files','-s','--','models/model3_khusboo/forecasting').splitlines()}
    binaries=[]
    for name in index:
        if Path(name).suffix in ('.joblib','.gz','.xlsx','.png'):
            data=(REPO/name).read_bytes()
            sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert sha==index[name],f'Raw bytes differ from staged Git object: {name}'
            binaries.append(name)
    assert not git('diff',BASE,'--name-only','--','.',':!models/model3_khusboo')
    record={'passed':True,'base_commit':BASE,'all_existing_files_outside_model3_unchanged':True,'historical_model3_artifacts_unchanged':True,'allowed_modified_existing_file':'models/model3_khusboo/README.md','binary_files_checked':len(binaries),'binary_bytes_match_git_objects':True}
    (ROOT/'results/git_scope_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()

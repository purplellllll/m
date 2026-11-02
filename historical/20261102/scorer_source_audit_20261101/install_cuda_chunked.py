"""Resume bounded HTTP ranges from the official wheel, then check its published hash."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
URL='https://download.pytorch.org/whl/cu126/torch-2.11.0%2Bcu126-cp312-cp312-win_amd64.whl'
SIZE=2596453924
SHA='1b7d728ddcf84b12f64ca10ec5f6ce91dd9edcc5e62ed5f0a7260ca00f029e8b'
CHUNK=16*1024*1024
CACHE=WORK/'.runtime/scorer_cuda_download'
PYTHON=WORK/'.runtime/scorer_cuda/Scripts/python.exe'

def status(stage,**fields):
    p=ROOT/'cuda_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,unix=time.time(),**fields),indent=2),encoding='utf-8')
    for attempt in range(20):
        try:tmp.replace(p);break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.1)

def download(index):
    start=index*CHUNK;end=min(SIZE,start+CHUNK)-1;length=end-start+1
    path=CACHE/f'{index:04d}.part'
    if path.exists() and path.stat().st_size==length:return path
    for attempt in range(8):
        try:
            request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=90) as response:
                assert response.status==206
                assert response.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}'
                data=response.read(length+1)
                assert len(data)==length
            temp=path.with_suffix('.tmp');temp.write_bytes(data);temp.replace(path)
            return path
        except Exception as error:
            print(f'Range {index}, attempt {attempt+1}: {type(error).__name__}',flush=True)
            if attempt==7:raise
            time.sleep(min(8,2**attempt))

def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    count=(SIZE+CHUNK-1)//CHUNK;status('downloading_verified_ranges',chunks=count,completed_chunks=0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done,path in enumerate(pool.map(download,range(count)),1):
            status('downloading_verified_ranges',chunks=count,completed_chunks=done,total_bytes=SIZE)
            if done%10==0:print(f'Downloaded {done}/{count} ranges',flush=True)
    status('checking_wheel_hash')
    wheel=CACHE/'torch-2.11.0+cu126-cp312-cp312-win_amd64.whl';digest=hashlib.sha256()
    with wheel.open('wb') as target:
        for index in range(count):
            data=(CACHE/f'{index:04d}.part').read_bytes();digest.update(data);target.write(data)
    assert wheel.stat().st_size==SIZE and digest.hexdigest()==SHA,'Official wheel SHA256 mismatch'
    status('installing_verified_wheel',sha256=SHA)
    subprocess.run([str(PYTHON),'-m','pip','install','--ignore-installed','--no-deps',str(wheel),'--disable-pip-version-check'],check=True)
    status('checking_gpu')
    for script in ['runtime_probe.py','length_probe.py']:
        subprocess.run([str(PYTHON),str(ROOT/script),'--device','cuda'],check=True,cwd=WORK)
    status('completed',python=str(PYTHON),wheel_sha256=SHA)

if __name__=='__main__':
    try:main()
    except BaseException as error:status('failed',error=repr(error));raise

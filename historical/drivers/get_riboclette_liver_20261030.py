"""Extract one public Ribo-seq dataset from a ZIP using HTTP byte ranges.

This is a related study's processed data, not an asserted RiboGL paper split.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import urllib.request
import zlib

ROOT=Path('/root/gpufree-data/ribogl_reproduction_20261030')
URL='https://zenodo.org/api/records/17100113/files/data.zip/content'
START=1421178532
SIZE=50248803
UNCOMPRESSED_SIZE=143460429
CRC32=0x41f01460
CHUNK=4*1024*1024


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)


def fetch(fd,index):
    local=index*CHUNK;n=min(CHUNK,SIZE-local);a=START+local;b=a+n-1
    h=hashlib.sha256();read=0
    # The endpoint truncated a 4 MiB response; smaller ranges avoid losing a full chunk.
    while read<n:
        piece_start=a+read;piece_end=min(b,piece_start+512*1024-1)
        request=urllib.request.Request(URL,headers={'Range':f'bytes={piece_start}-{piece_end}',
            'User-Agent':'RiboGL-academic-evaluation/1.0'})
        with urllib.request.urlopen(request,timeout=90) as r:
            assert r.status==206 and r.headers.get('Content-Range')==f'bytes {piece_start}-{piece_end}/2436029357'
            piece_read=0;piece_size=piece_end-piece_start+1
            while piece_read<piece_size:
                block=r.read(min(piece_size-piece_read,256*1024))
                if not block:raise IOError(f'incomplete range {index}, piece offset {piece_read}')
                h.update(block);done=0
                while done<len(block):
                    count=os.pwrite(fd,block[done:],local+read+done)
                    if count<=0:raise IOError('pwrite failed')
                    done+=count
                read+=len(block);piece_read+=len(block)
    return index,h.hexdigest()


def main(resume):
    os.nice(10);started=time.monotonic()
    p=ROOT/'related_liver';p.mkdir(exist_ok=True)
    partial=p/'liver_zip_member.deflate.part';receipt=p/'chunks.json';status=p/'status.json'
    def progress(stage,**kw):save(status,dict(stage=stage,elapsed=time.monotonic()-started,**kw))
    try:
        if resume:
            records=json.loads(receipt.read_text());complete={int(k):v for k,v in records.items()}
            fd=os.open(partial,os.O_RDWR)
            for i,h in complete.items():
                assert hashlib.sha256(os.pread(fd,min(CHUNK,SIZE-i*CHUNK),i*CHUNK)).hexdigest()==h
        else:
            assert not partial.exists() and not receipt.exists()
            complete={};fd=os.open(partial,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600);os.ftruncate(fd,SIZE)
        count=(SIZE+CHUNK-1)//CHUNK
        progress('downloading',completed=len(complete),total=count)
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(fetch,fd,i) for i in range(count) if i not in complete]
                try:
                    for future in as_completed(futures):
                        i,h=future.result();complete[i]=h;save(receipt,{str(k):v for k,v in complete.items()})
                        progress('downloading',completed=len(complete),total=count)
                except BaseException:
                    for future in futures:future.cancel()
                    raise
        finally:os.close(fd)
        assert len(complete)==count
        progress('extracting')
        target=p/'LIVER.csv';tmp=p/'LIVER.csv.part'
        assert not target.exists() and not tmp.exists()
        dec=zlib.decompressobj(-15);crc=0;length=0
        with partial.open('rb') as src,tmp.open('xb') as dst:
            while True:
                block=src.read(1024*1024)
                if not block:break
                output=dec.decompress(block);dst.write(output);crc=zlib.crc32(output,crc);length+=len(output)
            output=dec.flush();dst.write(output);crc=zlib.crc32(output,crc);length+=len(output)
        assert dec.eof and length==UNCOMPRESSED_SIZE and crc==CRC32,(dec.eof,length,hex(crc))
        tmp.replace(target)
        h=hashlib.sha256()
        with target.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        result=dict(source_url=URL,archive_record='10.5281/zenodo.17100113',member='data/processed/LIVER.csv',
            zip_member_crc32=hex(crc),uncompressed_size=length,sha256=h.hexdigest(),
            note='Related study processed GSE73553; source schema and split must be checked against RiboGL before use')
        save(p/'receipt.json',result);progress('completed',size=length,sha256=h.hexdigest())
        print(json.dumps(result,indent=2))
    except BaseException as exc:
        progress('failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');main(parser.parse_args().resume)

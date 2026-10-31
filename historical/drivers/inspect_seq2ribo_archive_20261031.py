"""List a remote ZIP with exact HTTP ranges; do not download its full contents."""
import io
import json
from pathlib import Path
import urllib.request
import zipfile

URL='https://zenodo.org/api/records/17100113/files/data.zip/content'
SIZE=2436029357
OUT=Path(__file__).resolve().parents[1]/'runs/scalar_downstream_20261031'


class RangedZip(io.RawIOBase):
    def __init__(self):self.position=0;self.total_read=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else SIZE+offset
        return self.position
    def read(self,size=-1):
        if size<0:size=SIZE-self.position
        size=min(size,SIZE-self.position)
        if not size:return b''
        assert size<=8*1024*1024,'Unexpectedly large ZIP directory request'
        start=self.position;end=start+size-1
        req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','User-Agent':'scalar-task-reproduction/1.0'})
        with urllib.request.urlopen(req,timeout=45) as response:
            assert response.status==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}'
            data=response.read(size+1);assert len(data)==size
        self.position+=size;self.total_read+=size;return data


def main():
    OUT.mkdir(exist_ok=True);remote=RangedZip()
    with zipfile.ZipFile(remote) as z:
        files=[dict(name=f.filename,size=f.file_size,compressed_size=f.compress_size,header_offset=f.header_offset,
            crc32=f.CRC,compression=f.compress_type) for f in z.infolist()]
    (OUT/'public_data_zip_listing.json').write_text(json.dumps(dict(url=URL,size=SIZE,bytes_read=remote.total_read,files=files),indent=2))
    print(json.dumps(dict(bytes_read=remote.total_read,n_files=len(files),interesting=[r for r in files if any(s in r['name'].lower() for s in ['mrfp','expr','train','test','te_','utr'])]),indent=2))


if __name__=='__main__':main()

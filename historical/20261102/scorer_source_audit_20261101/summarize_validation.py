"""Compare completed CPU/GPU checks; write a single independently checked receipt."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    assert read('cuda_status.json')['stage']=='completed'
    cpu_receipt=read('cpu_validation_complete.json');assert cpu_receipt['status']=='passed'
    for name,digest in cpu_receipt['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    cpu=read('runtime_probe_cpu.json');gpu=read('runtime_probe_cuda.json')
    lookup={(r['seed'],r['sequence_id']):r for r in cpu['records']};errors={}
    for row in gpu['records']:
        reference=lookup[row['seed'],row['sequence_id']]
        assert row['checkpoint_sha256']==reference['checkpoint_sha256']
        for key in ['baseline','without_pair_messages','diagonal_context','physics_zero','structure_zero']:
            error=abs(row[key]-reference[key]);errors[key]=max(errors.get(key,0.),error)
            assert error<5e-4,(key,error)
        assert row['physics_input_gradient_l1']>0 and row['structure_input_gradient_l1']>0
    length=read('length_probe_cuda.json')
    assert length['full_window_equivalence_max_abs_error']<2e-5
    assert all(row['max_abs_error']<2e-5 for row in length['local_mask_equivalence'])
    assert all(row.get('finite',True) for row in length['benchmarks'])
    receipt=dict(status='passed',unix=time.time(),cpu_torch=cpu['torch'],gpu_torch=gpu['torch'],gpu=gpu['gpu'],
                 compared_frozen_cases=len(gpu['records']),max_cpu_gpu_abs_error=errors,
                 benchmarks=length['benchmarks'],limits='Functional and synthetic inference checks; no new held-out accuracy or training-memory validation.')
    (ROOT/'validation_complete.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    lines=['# CPU与GPU验证完成','',f"GPU：{gpu['gpu']}；CPU PyTorch {cpu['torch']}，GPU PyTorch {gpu['torch']}。",'',
           'CPU物理入口、图门控、梯度、样本隔离和标量读出屏蔽检查通过。六组冻结输入的CPU/GPU输出误差最大为'+f"{max(errors.values()):.8g}"+'。','',
           '| 密码子长度 | 注意力 | 单次推理秒 | 峰值分配显存MiB |','|---:|---|---:|---:|']
    for row in length['benchmarks']:
        if row.get('status')=='out_of_memory':lines.append(f"| {row['n_codons']} | {row['mode']} | OOM | — |")
        else:lines.append(f"| {row['n_codons']} | {row['mode']} | {row['seconds']:.5f} | {row['peak_allocated_bytes']/2**20:.1f} |")
    lines+=['','以上是batch=1、width192、两层context的合成输入推理检查。为保持算术一致，使用FP32、禁用TF32、固定math SDPA并关闭两种实现的MHA/Transformer融合快速路径。该基线不是全注意力经过Flash等内核优化后的最佳性能，不能把本次显存比例泛化到所有全注意力实现。首轮GPU融合快速路径与普通实现差异约5.75e-4，统一路径后等价检查通过，未放宽阈值。','',
            '显存指PyTorch峰值分配，非整机显存；未包含完整RNA折叠/模拟、反向传播或训练优化器。滑窗原型未加入padding及全局摘要位置。可运行不表示长CDS预测准确。生产模型和CPU搜索协议未修改。']
    (ROOT/'CPU与GPU验证结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()

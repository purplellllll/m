"""Synchronize benchmark tables using an SSH password held only in memory."""
import argparse
import csv
import datetime
import getpass
import json
from pathlib import Path
import statistics
import time

import paramiko

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/rna_external_comparison_20261026'
REMOTE = '/root/gpufree-data/rna_external_comparison_20261026/'
FILES = ['summary.json','live_status.json','source_manifest.json','server_snapshot.json',
         'REPORT.md','metrics.csv','scoring_receipt.json','extra_scores/ribonn_scores.csv',
         'extra_scores/ribonn_excluded.json','extra_scores/ribonn_run_manifest.json']
LABELS = dict(original='天然序列',vmae971='VMAE 971万',vmae249='VMAE 250万',native='原生RiboDecode完整预算',
              codontransformer='CodonTransformer',fpp_codonbert='FPPGroup CodonBERT',gemorna='GEMORNA',
              lineardesign_l0='LinearDesign λ=0',lineardesign_l3='LinearDesign λ=3')


def report():
    summary = json.loads((OUT/'summary.json').read_text())
    live = json.loads((OUT/'live_status.json').read_text())
    stamp = datetime.datetime.fromtimestamp(live['heartbeat_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    text = '# 模型对比与服务器工作报告\n\n'
    text += '最近服务器状态：'+stamp+'。本文件由本地监视器更新；若时间不再推进，请核查监视器与服务器连接。\n\n'
    text += '250万版本已发布至 [GitHub V3](https://github.com/purplellllll/m/tree/V3/vmae_tiny_2498240_V3)，'
    text += '[最优权重](https://github.com/purplellllll/m/releases/tag/vmae-v3-2498240-20261026)已回下载核验。'
    text += '参数量2,498,240，最优epoch42，训练50轮；971万版参数量9,714,496，最优epoch35。'
    text += '两版都完成同一固定30条人类CDS评分。\n\n'
    text += 'RNop使用 [HudenJear/RPLoss](https://github.com/HudenJear/RPLoss)，源码已下载；作者Drive权重需要登录，按用户确认标记为不可获取，未产生RNop结果。\n\n'
    text += '## 运行情况\n\n'
    text += f"- GEMORNA：生成{live['generated'].get('gemorna',0)}/30，统一评分{live['scored'].get('gemorna',0)}/30。\n"
    for key in ['lineardesign_l0','lineardesign_l3']:
        text += f"- {LABELS[key]}：生成{live['generated'].get(key,0)}/30，统一评分{live['scored'].get(key,0)}/30。\n"
    text += f"- 原生RiboDecode完整20轮预算：{live['full_native_completed']}/30。未完成的条目继续运行，不替换成低预算结果。\n"
    text += '- 服务器另一项250万参数多目标微调与评估任务原样保留，详见server_snapshot.json的采集时间和状态。\n'
    text += '- 新任务目录：`/root/gpufree-data/rna_external_comparison_20261026`；状态见live_status.json，逐条候选见generated/，指标见metrics/。\n\n'
    if live['errors']:
        text += '**需处理：'+ '; '.join(live['errors'])+'**\n\n'
    text += '## 统一指标\n\n下表仅作阶段记录。N不同的行不能直接排名，配对比较必须使用相同CDS身份；30条完整结论只用于N=30的方法。\n\n'
    keys = ['ribodecode_score','seq2ribo_score','cai','gc_percent','mfe_kcal_mol','mfe_per_nt']
    text += '| 方法 | N | RiboDecode | seq2ribo | CAI | GC % | MFE kcal/mol | MFE/nt |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for name in LABELS:
        if name in summary['methods']:
            v = summary['methods'][name]
            text += '| '+LABELS[name]+' | '+str(v['n'])+' | '+' | '.join(f"{v['means'][k]:.4f}" for k in keys)+' |\n'
    text += '\n## 与两个VMAE的相同CDS配对\n\n差值为外部方法减VMAE；指标尺度互不通用。\n\n'
    text += '| 外部方法 | 参考 | 配对N | ΔRiboDecode | Δseq2ribo | ΔCAI | ΔGC百分点 | ΔMFE |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for method in ['gemorna','lineardesign_l0','lineardesign_l3']:
        for reference in ['vmae971','vmae249']:
            v = summary.get('paired',{}).get(method,{}).get(reference)
            if v:
                text += '| '+LABELS[method]+' | '+LABELS[reference]+' | '+str(v['n'])+' | '+' | '.join(f"{v['mean_delta'][k]:+.4f}" for k in keys[:-1])+' |\n'
    path = OUT/'extra_scores/ribonn_scores.csv'
    if path.exists():
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        text += '\n## 补充RiboNN评分\n\n使用相同天然UTR上下文、官方50模型集成；全长超限的1455aa条目保留缺失。N不同仍须配对。\n\n'
        text += '| 方法 | N | HEK293 | HEK293T | 78背景均分 |\n|---|---:|---:|---:|---:|\n'
        for name in LABELS:
            selected = [r for r in rows if r['method']==name]
            if selected:
                text += '| '+LABELS[name]+' | '+str(len(selected))+' | '+' | '.join(f'{statistics.mean(float(r[k]) for r in selected):.4f}' for k in ['predicted_TE_HEK293','predicted_TE_HEK293T','ribonn_mean_predicted_TE'])+' |\n'
    text += '\n## 口径与可解释范围\n\n'
    text += '- “CG”按GC含量处理。CAI由固定训练集频率计算；MFE统一ViennaRNA2.7.2、37℃、dangles2，全长和每nt同时报告。\n'
    text += '- RiboDecode保留HEK293T背景、丰度固定4.5；seq2ribo为HEK293 CDS-only TE，固定3种子，每种子8次sTASEP；候选生成后才做独立评分。\n'
    text += '- LinearDesign保留官方输出的原生MFE/人类表CAI，另算统一指标；默认λ=0，预定λ=3作为补充，不按测试分数调λ。\n'
    text += '- GEMORNA CDS生成器没有独立CDS表达评分头；其UTR预测器不适用于CDS-only面板。RNop机制损失不能冒充已运行的表达评分器。VMAE重建CE也不是表达评分。\n'
    text += '- 两版VMAE用80步潜空间搜索，原生RiboDecode为完整20×40000候选，搜索预算不相同。保留逐条时间与查询字段，不作等预算胜出结论。\n'
    text += '- 预测分数不能替代实测表达。GC/MFE没有普遍的单调优劣方向；外部模型训练集与本面板重叠尚未认证。\n'
    temp = OUT/'服务器与模型对比报告.md.tmp'
    temp.write_text(text,encoding='utf-8')
    temp.replace(OUT/'服务器与模型对比报告.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    password = getpass.getpass('SSH password: ')
    OUT.mkdir(parents=True,exist_ok=True)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known = Path.home()/'.ssh/known_hosts'
    if known.exists():
        client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        while True:
            try:
                if not client.get_transport() or not client.get_transport().is_active():
                    client.connect('183.147.142.40',port=31026,username='root',password=password,
                        look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20)
                    client.get_transport().set_keepalive(20)
                with client.open_sftp() as sftp:
                    for name in FILES:
                        local = OUT/name
                        local.parent.mkdir(parents=True,exist_ok=True)
                        temp = local.with_suffix('.download')
                        try:
                            sftp.get(REMOTE+name,str(temp))
                            temp.replace(local)
                        except FileNotFoundError:
                            pass
                report()
                live = json.loads((OUT/'live_status.json').read_text())
                print(json.dumps(dict(state=live['state'],scored=live['scored'],heartbeat=live['heartbeat_unix'])),flush=True)
                if args.once or live['state'] in ('completed','failed'):
                    break
            except (OSError,paramiko.SSHException) as exc:
                print(type(exc).__name__+': '+str(exc),flush=True)
                client.close()
                if args.once:
                    raise
            time.sleep(60)
    finally:
        client.close()


if __name__ == '__main__':
    main()

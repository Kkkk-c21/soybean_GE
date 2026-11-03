# -*- coding: utf-8 -*-
"""
读取VCF文件中的基因型数据，并将其转换为0/1/2编码的NumPy数组。
每个样本的基因型数据将被保存为单独的.npy文件，便于后续分析。
同时生成数据检查报告：SNP数、样本数、基因型分布、缺失率等。

编码规则：
    0/0, 0|0            -> 0  (纯合参考型)
    0/1,1/0,0|1,1|0 及
    0/2,2/0,1/2,2/1,
    1|2,2|1            -> 1  (杂合)
    1/1,2/2,1|1,2|2    -> 2  (纯合变异型)
    ./. , .|.           -> 0  (缺失，编码为0并单独计数)

用法：
    cd <项目根目录>
    python read_vcf_nan.py
"""
import csv
import os
import re
import gzip
import json
import multiprocessing as mp
import numpy as np
import pandas as pd

# =================== 路径配置 ===================
# 脚本就在这个目录里
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# → /Data4/gly_wkdir/soybean_agriculture/张洛伊/数据检查Dataset制作数据划分

# 输入路径
GENEOTYPE_PATH = '/Data4/gly_wkdir/soybean_agriculture/大豆vcf_环境信息/基因/extracted_final.vcf.gz'

# 输出目录：就用脚本所在目录本身
OUTPUT_ROOT = SCRIPT_DIR
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, 'numpy')     # 想不想要 numpy 由你决定
REPORT_PATH = os.path.join(OUTPUT_ROOT, 'vcf_data_check_report.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("输入文件路径:", GENEOTYPE_PATH)
print("报告输出路径:", REPORT_PATH)
print("数据输出目录:", OUTPUT_DIR)
# ================================================

csv.field_size_limit(500 * 1024 * 1024)


def load_vcf(path):
    """读取VCF，跳过##注释行，返回(样本名列表, 基因型DataFrame[样本 x SNP])"""
    skipped = []
    with gzip.open(path, 'rt') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if row[0].strip().startswith('##'):
                skipped.append(i)

    df = pd.read_csv(path, sep=r"\s+", skiprows=skipped, header=0)
    # 只保留SNP ID与样本基因型列
    sample_cols = [c for c in df.columns
                   if c not in ('#CHROM', 'POS', 'ID', 'REF', 'ALT',
                                'QUAL', 'FILTER', 'INFO', 'FORMAT')]
    df = df[['ID'] + sample_cols].set_index('ID')
    df = df.transpose()  # 行=样本, 列=SNP
    return list(df.index), df


def encode_sample(dataframe):
    """将单个样本的基因型行编码为uint8数组，返回(样本名, 数组, 缺失数)"""
    sample = dataframe.name
    values = dataframe.to_numpy(dtype=str)
    n = len(values)
    out = np.zeros(n, dtype=np.uint8)
    missing = 0
    for i, gt in enumerate(values):
        if gt in ('0/0', '0|0'):
            out[i] = 0
        elif gt in ('0/1', '1/0', '0|1', '1|0',
                    '0/2', '2/0', '1/2', '2/1',
                    '1|2', '2|1'):
            out[i] = 1
        elif gt in ('1/1', '2/2', '1|1', '2|2'):
            out[i] = 2
        elif gt in ('./.', '.|.'):
            out[i] = 0
            missing += 1
        else:
            raise ValueError(f"未知基因型：{gt}，样本：{sample}，SNP：{i}")
    return sample, out, missing


def main():
    print("=" * 60)
    print("步骤1：读取VCF并做格式检查")
    print("=" * 60)
    samples, df = load_vcf(GENEOTYPE_PATH)
    n_snp = df.shape[1]
    n_samples = df.shape[0]
    print(f"SNP数：{n_snp}")
    print(f"样本数：{n_samples}")

    # ---- 基因型取值分布检查 ----
    uniq = pd.unique(df.values.ravel())
    print("全部基因型取值：", sorted([str(u) for u in uniq]))
    pattern = re.compile(r'^[0-9.][/|][0-9.]$')
    bad = ~df.map(lambda x: bool(pattern.match(str(x))))
    if bad.any().any():
        raise AssertionError("存在不符合 ^[0-9.]/[0-9.]$ 格式的基因型！")
    print("格式检查通过：所有基因型均为 x/x 或 x|x 格式")

    # ---- 缺失率检查（SNP x 样本 层面）----
    mat = df.to_numpy(dtype=str)
    missing_mask = (mat == './.') | (mat == '.|.')
    overall_missing_rate = float(missing_mask.mean())
    per_snp_missing = missing_mask.mean(axis=0)   # 每个SNP缺失率
    per_sample_missing = missing_mask.mean(axis=1)  # 每个样本缺失率
    print(f"总体缺失率：{overall_missing_rate:.4%}")
    print(f"SNP缺失率范围：[{per_snp_missing.min():.4%}, {per_snp_missing.max():.4%}]")
    print(f"样本缺失率范围：[{per_sample_missing.min():.4%}, {per_sample_missing.max():.4%}]")

    print("\n" + "=" * 60)
    print("步骤2：多进程编码所有样本 -> 0/1/2")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_proc = min(32, mp.cpu_count())
    print(f"使用进程数：{n_proc}")

    with mp.Pool(processes=n_proc) as pool:
        results = pool.map(encode_sample, [df.loc[s] for s in samples])

    missing_counts = {}
    for sample, arr, miss in results:
        sample_name = str(sample).replace('/', '.')
        np.save(os.path.join(OUTPUT_DIR, sample_name + '.npy'), arr)
        missing_counts[sample_name] = miss

    print("\n========== 全部样本统计 ==========")
    total_missing = sum(missing_counts.values())
    print("样本总数：", len(results))
    print("缺失基因型总数：", total_missing)
    print(f"总体缺失率：{total_missing / (n_samples * n_snp):.4%}")

    # ---- 样本编码检查（第一个样本）----
    first_sample = str(samples[0]).replace('/', '.')
    a = np.load(os.path.join(OUTPUT_DIR, first_sample + '.npy'))
    print("\n========== 样本 {} 编码检查 ==========".format(first_sample))
    print("shape:", a.shape)
    print("数据类型:", a.dtype)
    print("前10个编码:", a[:10].tolist())
    counts = np.bincount(a, minlength=3)
    miss = missing_counts[first_sample]
    true_00 = int(counts[0]) - miss
    print("真实0/0数量:", true_00)
    print("杂合数量:", counts[1])
    print("纯合变异数量:", counts[2])
    print("缺失数量:", miss)
    assert true_00 + counts[1] + counts[2] + miss == len(a), "SNP数量统计不一致！"

    # ---- 前10个SNP编码对照 ----
    sample_row = df.loc[samples[0]]
    print("\n=== 样本 {} 前10个SNP编码检查 ===".format(first_sample))
    for i in range(min(10, len(sample_row))):
        snp_id = sample_row.index[i]
        print(f"SNP{i + 1:02d} | ID={snp_id} | GT={sample_row.iloc[i]} | 编码={int(a[i])}")

    # ---- 保存检查报告 ----
    report = {
        "vcf": GENEOTYPE_PATH,
        "n_snp": int(n_snp),
        "n_samples": int(n_samples),
        "genotype_types": sorted([str(u) for u in uniq]),
        "overall_missing_rate": overall_missing_rate,
        "total_missing": int(total_missing),
        "per_snp_missing_min": float(per_snp_missing.min()),
        "per_snp_missing_max": float(per_snp_missing.max()),
        "per_sample_missing_min": float(per_sample_missing.min()),
        "per_sample_missing_max": float(per_sample_missing.max()),
        "npy_output_dir": OUTPUT_DIR,
        "missing_counts_sample_234": missing_counts.get('234', None),
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n数据检查报告已保存：", REPORT_PATH)
    print("\n全部完成！npy文件输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()

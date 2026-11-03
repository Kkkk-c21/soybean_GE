# -*- coding: utf-8 -*-
"""
数据集划分脚本：按环境(地点)划分训练集/验证集，比例 训练:验证 = 5:2。

划分原则：
1. 划分单位是【地点】(env_code 去掉前4位年份后的部分)，
   同一地点不同年份(如 2022GC 与 2024GC 都是 GC)必须放在同一个集合中，绝不拆开。
2. 训练:验证 = 5:2 指【样本记录数】比例（训练约5/7，验证约2/7）。
   由于划分单位是地点，无法做到精确比例，
   故穷举所有地点子集作为验证集，选出验证集记录数最接近 total*2/7 的组合。

产出：
    <OUTPUT_DIR>/train.csv           训练集表型记录（含row_index原始行号）
    <OUTPUT_DIR>/val.csv             验证集表型记录（含row_index原始行号）
    <OUTPUT_DIR>/split_info.json     划分信息（比例、种子、表型均值/标准差等）
    <OUTPUT_DIR>/split_locations.json 地点归属明细

用法：
    cd <项目根目录>
    python split_dataset.py
"""
import os
import json
from itertools import combinations
import numpy as np
import pandas as pd

# =================== 参数配置 ===================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录（soybean_agriculture）
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  
# SCRIPT_DIR           = .../soybean_agriculture/张洛伊/数据检查Dataset制作数据划分
# dirname(SCRIPT_DIR)  = .../soybean_agriculture/张洛伊
# dirname 再上一层      = .../soybean_agriculture

# ============ 输入：刚生成的 npy ============
# 脚本同目录下的 numpy/ 文件夹
GENOTYPE_DIR = os.path.join(SCRIPT_DIR, 'numpy')

# ============ 输入：表型 / 环境（仍在 大豆vcf_环境信息/ 里） ============
DATA_ROOT = os.path.join(PROJECT_ROOT, '大豆vcf_环境信息')
# 注意：你原来写的是 '大豆vcf&环境信息'（& 应该是笔误），实际文件夹名是 '大豆vcf_环境信息'

PHENOTYPE_FILE = os.path.join(DATA_ROOT, '表型', 'Phenotype_pre',
                              '7envOIL_phenotype_filtered.txt')
ENV_FILE = os.path.join(DATA_ROOT, '环境数据', 'Environment_18_Normalization',
                        'Envs_envParas_DAP100_OIL_norm.txt')

TRAIT = 'OIL'

# 训练:验证 = 5:2
TRAIN_PART, VAL_PART = 5, 2
VAL_RATIO = VAL_PART / (TRAIN_PART + VAL_PART)   # 2/7 ≈ 0.2857

SEED = 40
BATCH_SIZE = 32
SELECTED_ENV_COLS = None   # None 表示使用环境文件全部环境因子

# ============ 输出：仍然落在当前脚本目录里 ============
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '划分结果', f'environment_{TRAIT}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ================================================


def load_phenotype():
    """读取表型文件并做完整性检查"""
    table = pd.read_csv(PHENOTYPE_FILE, sep='\t')
    for col in ('line_code', 'env_code', TRAIT):
        assert col in table.columns, f'表型文件缺少列: {col}'
    table['line_code'] = table['line_code'].astype(str)
    table['env_code'] = table['env_code'].astype(str)

    print('========== 表型数据检查 ==========')
    print('总记录数：', len(table))
    print('材料数：', table['line_code'].nunique())
    print('环境数：', table['env_code'].nunique())
    print('各环境记录数：')
    print(table['env_code'].value_counts().to_string())
    print('表型缺失数：', table[TRAIT].isna().sum())
    assert not table[TRAIT].isna().any(), '存在缺失表型，先核对原数据'
    assert np.isfinite(table[TRAIT].to_numpy(dtype=float)).all(), \
        '表型包含 NaN 或无穷大'
    assert not table.duplicated(['line_code', 'env_code']).any(), \
        '存在 材料-环境 重复记录'
    return table


def check_genotype_files(table):
    """检查表型中每个材料都有对应 npy 编码文件"""
    npy_names = {os.path.splitext(f)[0]
                 for f in os.listdir(GENOTYPE_DIR) if f.endswith('.npy')}
    lines = set(table['line_code'])
    missing = sorted(lines - npy_names)
    assert not missing, f'缺少基因型编码文件：{missing}'
    extra = sorted(npy_names - lines)
    print(f'基因型-表型匹配检查：{len(lines)} 个材料全部有 npy 文件')
    if extra:
        print(f'注意：{len(extra)} 个基因型样本无表型记录（不影响建模）：{extra}')


def check_env_file(table):
    """检查环境数据文件覆盖所有环境"""
    envs = pd.read_csv(ENV_FILE, sep=r'\s+')
    env_codes = set(envs['env_code'].astype(str))
    missing_env = set(table['env_code']) - env_codes
    assert not missing_env, f'环境数据缺少：{missing_env}'
    print(f'环境数据检查：覆盖 {len(env_codes)} 个环境，'
          f'每个环境 {envs.groupby("env_code").size().iloc[0]} 天')
    return envs


def make_location_split(table, target_val):
    """
    按地点划分，穷举所有地点子集作为验证集，
    选验证集记录数最接近 target_val 的组合。
    返回 (train_locations, val_locations, 验证记录数)
    """
    table = table.copy()
    table['location'] = table['env_code'].str[4:]   # 2022GC -> GC
    loc_size = table.groupby('location').size().to_dict()
    locations = sorted(loc_size.keys())
    total = len(table)

    print('\n========== 地点分组 ==========')
    for loc in locations:
        envs = sorted(table.loc[table['location'] == loc, 'env_code'].unique())
        print(f'地点 {loc}: {loc_size[loc]} 条记录, 环境 {envs}')

    best = None  # (diff, n_val, combo)
    for k in range(1, len(locations)):
        for combo in combinations(locations, k):
            n_val = sum(loc_size[loc] for loc in combo)
            diff = abs(n_val - target_val)
            if best is None or diff < best[0]:
                best = (diff, n_val, combo)

    _, n_val, val_locs = best
    val_locs = list(val_locs)
    train_locs = [loc for loc in locations if loc not in val_locs]
    n_train = total - n_val

    print(f'\n目标验证集记录数(total*{VAL_RATIO:.4f})：{target_val:.1f}')
    print(f'最优验证地点：{val_locs}（{n_val} 条记录）')
    print(f'训练地点：{train_locs}（{n_train} 条记录）')
    print(f'实际比例 训练:验证 = {n_train}:{n_val} '
          f'= {n_train / total:.4f}:{n_val / total:.4f} '
          f'（目标 5:2 = 0.7143:0.2857）')
    return train_locs, val_locs


def build_ids(table, train_locs, val_locs):
    """按地点生成训练/验证记录行号"""
    table = table.copy()
    table['location'] = table['env_code'].str[4:]
    table['row_index'] = range(len(table))
    train_ids = list(table.loc[table['location'].isin(train_locs), 'row_index'])
    val_ids = list(table.loc[table['location'].isin(val_locs), 'row_index'])

    # 完整性校验
    assert len(set(train_ids) & set(val_ids)) == 0, '训练/验证存在重叠记录'
    assert len(train_ids) + len(val_ids) == len(table), '存在未分配记录'
    # 同地点不拆分校验
    train_locs_set = set(train_locs)
    val_locs_set = set(val_locs)
    assert not (train_locs_set & val_locs_set), '同一地点被拆到两个集合'
    # GC 专项检查（2022GC 与 2024GC 必须同集合）
    train_envs = set(table.iloc[train_ids]['env_code'])
    val_envs = set(table.iloc[val_ids]['env_code'])
    gc = {'2022GC', '2024GC'}
    assert not (gc & train_envs and gc & val_envs), \
        '2022GC 与 2024GC 被划分到了不同集合！'
    print('校验通过：地点完整、无重叠、2022GC/2024GC 同属', 
          '训练集' if gc <= train_envs else '验证集')
    return train_ids, val_ids


def save_split(table, train_ids, val_ids):
    """保存划分结果"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, ids in (('train', train_ids), ('val', val_ids)):
        part = table.iloc[ids].copy()
        part.insert(0, 'row_index', ids)
        out = os.path.join(OUTPUT_DIR, name + '.csv')
        part.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'{name}.csv: 记录数={len(part)}, 材料数={part["line_code"].nunique()}, '
              f'环境数={part["env_code"].nunique()}')

    train_y = table.iloc[train_ids][TRAIT].to_numpy(dtype=float)
    val_y = table.iloc[val_ids][TRAIT].to_numpy(dtype=float)
    info = {
        'trait': TRAIT,
        'split_mode': 'environment(by location)',
        'note': '同一地点不同年份(如2022GC/2024GC)始终在同一数据集合',
        'train_val_ratio': f'{TRAIN_PART}:{VAL_PART}',
        'val_ratio_expected': VAL_RATIO,
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'n_total': len(table),
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'train_ratio_actual': round(len(train_ids) / len(table), 4),
        'val_ratio_actual': round(len(val_ids) / len(table), 4),
        'train_y_mean': float(np.mean(train_y)),
        'train_y_std': float(np.std(train_y)),
        'val_y_mean': float(np.mean(val_y)),
        'val_y_std': float(np.std(val_y)),
    }
    with open(os.path.join(OUTPUT_DIR, 'split_info.json'), 'w',
              encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print('\nsplit_info.json 已保存。')


def main():
    table = load_phenotype()
    check_genotype_files(table)
    check_env_file(table)

    total = len(table)
    target_val = total * VAL_RATIO
    train_locs, val_locs = make_location_split(table, target_val)
    train_ids, val_ids = build_ids(table, train_locs, val_locs)
    save_split(table, train_ids, val_ids)

    print('\n========== 数据划分结果 ==========')
    print('划分方式：按地点(environment)')
    print('训练集记录数：', len(train_ids))
    print('验证集记录数：', len(val_ids))
    print('训练集比例：', round(len(train_ids) / total, 4))
    print('验证集比例：', round(len(val_ids) / total, 4))
    print('输出目录：', OUTPUT_DIR)


if __name__ == '__main__':
    main()

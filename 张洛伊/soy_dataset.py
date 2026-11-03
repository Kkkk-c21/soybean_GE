# -*- coding: utf-8 -*-
"""
读取划分结果文件，构建 SoyDataset 与 DataLoader，供后续模型训练使用。

每个样本由三部分组成：
    g: 基因型编码向量 (n_snp,)            <- numpy/<line_code>.npy (0/1/2)
    e: 环境时间序列   (n_days, n_factors) <- 环境norm文件 (如 100天 x 16因子)
    y: 表型标量       (1,)               <- 表型文件 OIL 列

用法：
    cd <项目根目录>
    python soy_dataset.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# =================== 路径配置（与 split_dataset.py 保持一致）===================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据根目录（表型 / 环境 / VCF 都在这里）
# 注意：实际文件夹名是 "大豆vcf_环境信息"，不是 "大豆vcf&环境信息"
DATA_ROOT = '/Data4/gly_wkdir/soybean_agriculture/大豆vcf_环境信息'

# ---- 输入：基因型 npy（上一步在脚本同目录的 numpy/ 下生成）----
GENOTYPE_DIR = os.path.join(SCRIPT_DIR, 'numpy')

# ---- 输入：表型 / 环境（仍在 大豆vcf_环境信息/ 下）----
PHENOTYPE_FILE = os.path.join(DATA_ROOT, '表型', 'Phenotype_pre',
                              '7envOIL_phenotype_filtered.txt')
ENV_FILE = os.path.join(DATA_ROOT, '环境数据', 'Environment_18_Normalization',
                        'Envs_envParas_DAP100_OIL_norm.txt')

# ---- 输出：与 split_dataset.py 保持一致，落在脚本当前目录 ----
SPLIT_DIR = os.path.join(SCRIPT_DIR, '划分结果', 'environment_OIL')

TRAIT = 'OIL'
BATCH_SIZE = 32
# 环境因子列；None 表示自动取 Date 之后、lat 之前的全部数值列
SELECTED_ENV_COLS = None

# 确保输出目录存在
os.makedirs(SPLIT_DIR, exist_ok=True)
# =============================================================================

# 环境norm文件固定结构：env_code Date [因子...] lat lon
DEFAULT_ENV_FACTORS = ['TMAX', 'TMIN', 'DL', 'GDD', 'dGDD', 'DTR',
                       'PTT', 'PTR', 'PTD1', 'PTD2', 'TSR', 'MMR',
                       'PR', 'RH', 'PRDTR', 'dPTT']


class SoyDataset(Dataset):
    """大豆基因型-环境-表型数据集"""

    def __init__(self, phenotype_file, genotype_dir, env_file,
                 trait=TRAIT, selected_env_cols=None):
        # ---- 表型 ----
        self.records = pd.read_csv(phenotype_file, sep='\t')
        self.records['line_code'] = self.records['line_code'].astype(str)
        self.records['env_code'] = self.records['env_code'].astype(str)
        assert trait in self.records.columns, f'表型缺少列 {trait}'
        assert not self.records[trait].isna().any(), '表型存在缺失'

        # ---- 基因型目录 ----
        self.genotype_dir = genotype_dir

        # ---- 环境数据：按 env_code 分组为 (日期 x 因子) 矩阵 ----
        env = pd.read_csv(env_file, sep=r'\s+')
        env['env_code'] = env['env_code'].astype(str)
        if selected_env_cols is None:
            factors = [c for c in DEFAULT_ENV_FACTORS if c in env.columns]
        else:
            factors = [c for c in selected_env_cols if c in env.columns]
        assert factors, '未找到任何环境因子列'
        self.factors = factors
        self.env_table = {}
        for code, grp in env.groupby('env_code'):
            grp = grp.sort_values('Date')
            self.env_table[code] = grp[factors].to_numpy(dtype=np.float32)

        # 预计算行数
        self._cache_g = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        line, env_code = row['line_code'], row['env_code']

        # g: 基因型 (n_snp,)
        if line not in self._cache_g:
            arr = np.load(os.path.join(self.genotype_dir, line + '.npy'))
            self._cache_g[line] = arr.astype(np.float32)
        g = torch.from_numpy(self._cache_g[line])

        # e: 环境 (n_days, n_factors)
        e = torch.from_numpy(self.env_table[env_code])

        # y: 表型标量
        y = torch.tensor(float(row[TRAIT]), dtype=torch.float32)
        return g, e, y


def main():
    dataset = SoyDataset(PHENOTYPE_FILE, GENOTYPE_DIR, ENV_FILE,
                         trait=TRAIT, selected_env_cols=SELECTED_ENV_COLS)

    print('========== 数据集构建 ==========')
    print('总记录数：', len(dataset))
    print('材料数：', dataset.records['line_code'].nunique())
    print('环境数：', dataset.records['env_code'].nunique())
    print('环境因子：', dataset.factors)
    print('环境序列天数 x 因子数：',
          dataset.env_table[dataset.records['env_code'].iloc[0]].shape)

    # 读取划分结果
    train_df = pd.read_csv(os.path.join(SPLIT_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(SPLIT_DIR, 'val.csv'))
    train_ids = train_df['row_index'].tolist()
    val_ids = val_df['row_index'].tolist()
    print(f'划分读取：train={len(train_ids)} val={len(val_ids)} '
          f'(来自 {SPLIT_DIR})')

    # 构建 DataLoader
    train_loader = DataLoader(Subset(dataset, train_ids),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_ids),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    # 验证一个 batch
    g, e, y = next(iter(train_loader))
    print('\n========== DataLoader 验证 ==========')
    print('一个训练 batch：')
    print('  g (基因型):', tuple(g.shape), g.dtype)
    print('  e (环境):  ', tuple(e.shape), e.dtype)
    print('  y (表型):  ', tuple(y.shape), y.dtype)
    print('  y 数值范围: [{:.3f}, {:.3f}]'.format(y.min().item(), y.max().item()))

    g, e, y = next(iter(val_loader))
    print('一个验证 batch：')
    print('  g (基因型):', tuple(g.shape))
    print('  e (环境):  ', tuple(e.shape))
    print('  y (表型):  ', tuple(y.shape))

    # 保存划分信息摘要供模型训练使用
    with open(os.path.join(SPLIT_DIR, 'split_info.json'), 'r',
              encoding='utf-8') as f:
        info = json.load(f)
    print('\n========== 划分信息摘要 ==========')
    for k in ('trait', 'split_mode', 'train_val_ratio', 'n_train', 'n_val',
              'train_ratio_actual', 'val_ratio_actual',
              'train_y_mean', 'train_y_std'):
        print(f'  {k}: {info.get(k)}')
    print('\nDataLoader 构建成功，可直接用于模型训练！')
    return train_loader, val_loader


if __name__ == '__main__':
    main()

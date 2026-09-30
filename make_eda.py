import os, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ZIP = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五\赛题五\赛题五数据.zip'
OUT = r'E:\ChatGPT\ChatGPT_Project\深圳金融比赛赛题五\eda_outputs'
os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 220
plt.rcParams['axes.titleweight'] = 'bold'

z = zipfile.ZipFile(ZIP)
train_daily = []
test_daily = []
train_y = []
train_amt = []; test_amt = []
train_vol = []; test_vol = []
train_close = []; test_close = []
flag_rows = []
missing_rows = []

for name in ['训练集.csv', '测试集_X.csv']:
    is_train = name.startswith('训练')
    with z.open(name) as f:
        for c in pd.read_csv(f, chunksize=250000):
            feat = ['open','high','low','close','vol','amount']
            miss = c[feat].isna().any(axis=1)
            cc = c.assign(_miss=miss)
            g = cc.groupby('trade_date').agg(
                miss_sum=('_miss','sum'), n=('trade_date','size'),
                up_sum=('flag_limit_up','sum'), down_sum=('flag_limit_down','sum'))
            if is_train:
                cv = c.dropna(subset=['y_ret_1d'])
                gy = cv.assign(ret_sq=cv.y_ret_1d**2).groupby('trade_date').agg(ret_sum=('y_ret_1d','sum'), ret_sq_sum=('ret_sq','sum'), ret_n=('y_ret_1d','size'))
                g = g.join(gy)
                train_daily.append(g)
                train_y.append(c.y_ret_1d.dropna().sample(min(150000, c.y_ret_1d.notna().sum()), random_state=42))
                for dest, col in [(train_amt,'amount'),(train_vol,'vol'),(train_close,'close')]:
                    dest.append(c[col].dropna().sample(min(150000, c[col].notna().sum()), random_state=42).to_numpy())
            else:
                test_daily.append(g)
                for dest, col in [(test_amt,'amount'),(test_vol,'vol'),(test_close,'close')]:
                    dest.append(c[col].dropna().sample(min(150000, c[col].notna().sum()), random_state=42).to_numpy())

trd = pd.concat(train_daily).groupby(level=0).sum().sort_index()
ted = pd.concat(test_daily).groupby(level=0).sum().sort_index()
for d in [trd, ted]:
    d['miss_rate'] = d.miss_sum / d.n
    d['limit_up'] = d.up_sum / d.n
    d['limit_down'] = d.down_sum / d.n
trd['ret_mean'] = trd.ret_sum / trd.ret_n
y = pd.concat(train_y, ignore_index=True).to_numpy()
def cat(parts): return np.concatenate(parts)
tr_amt, te_amt = cat(train_amt), cat(test_amt)
tr_vol, te_vol = cat(train_vol), cat(test_vol)
tr_close, te_close = cat(train_close), cat(test_close)

COLORS = {'train':'#2563eb','test':'#f97316','green':'#16a34a','red':'#dc2626','gray':'#64748b'}

# 1. Overview
fig, ax = plt.subplots(2,3, figsize=(16,9), constrained_layout=True)
ax[0,0].hist(np.clip(y, -0.15, 0.15), bins=100, color=COLORS['train'], alpha=.88)
ax[0,0].axvline(np.mean(y), color=COLORS['red'], lw=1.5, label=f'均值 {np.mean(y):.3%}')
ax[0,0].axvline(np.median(y), color=COLORS['green'], lw=1.5, ls='--', label=f'中位数 {np.median(y):.3%}')
ax[0,0].set_title('训练集下一日收益率分布（截断显示）'); ax[0,0].set_xlabel('y_ret_1d'); ax[0,0].set_ylabel('样本数'); ax[0,0].legend(frameon=False)
miss = [trd.miss_rate.mean(), ted.miss_rate.mean()]
ax[0,1].bar(['训练集','测试集'], miss, color=[COLORS['train'],COLORS['test']]); ax[0,1].yaxis.set_major_formatter(PercentFormatter(1)); ax[0,1].set_title('量价字段缺失率'); ax[0,1].set_ylabel('缺失率')
for i,v in enumerate(miss): ax[0,1].text(i,v+0.004,f'{v:.2%}',ha='center')
ax[0,2].plot(trd.index, trd.ret_mean.rolling(20,min_periods=5).mean(), color=COLORS['train'], label='20日滚动收益均值')
ax[0,2].axhline(0,color='gray',lw=.8); ax[0,2].set_title('训练期横截面平均收益趋势'); ax[0,2].set_xlabel('交易日'); ax[0,2].set_ylabel('收益率'); ax[0,2].legend(frameon=False)
ax[1,0].plot(trd.index, trd.miss_rate, color=COLORS['train'], label='训练集'); ax[1,0].plot(ted.index, ted.miss_rate, color=COLORS['test'], label='测试集'); ax[1,0].yaxis.set_major_formatter(PercentFormatter(1)); ax[1,0].set_title('每日量价缺失比例'); ax[1,0].set_xlabel('交易日'); ax[1,0].set_ylabel('缺失率'); ax[1,0].legend(frameon=False)
bins=np.linspace(0, 26, 80)
ax[1,1].hist(np.log10(tr_amt+1), bins=bins, alpha=.65, density=True, color=COLORS['train'], label='训练集'); ax[1,1].hist(np.log10(te_amt+1), bins=bins, alpha=.65, density=True, color=COLORS['test'], label='测试集'); ax[1,1].set_title('成交额分布对比'); ax[1,1].set_xlabel('log10(amount + 1)'); ax[1,1].set_ylabel('密度'); ax[1,1].legend(frameon=False)
up=[trd.limit_up.mean(),ted.limit_up.mean()]; down=[trd.limit_down.mean(),ted.limit_down.mean()]
x=np.arange(2); w=.35; ax[1,2].bar(x-w/2,up,w,label='涨停',color=COLORS['red']); ax[1,2].bar(x+w/2,down,w,label='跌停',color=COLORS['green']); ax[1,2].set_xticks(x,['训练集','测试集']); ax[1,2].yaxis.set_major_formatter(PercentFormatter(1)); ax[1,2].set_title('涨跌停比例'); ax[1,2].set_ylabel('记录比例'); ax[1,2].legend(frameon=False)
fig.suptitle('股票收益预测赛题：初步 EDA 总览', fontsize=16)
fig.savefig(os.path.join(OUT,'eda_overview.png'), bbox_inches='tight'); plt.close(fig)

# 2. Returns detail
fig, ax = plt.subplots(1,2, figsize=(13,5), constrained_layout=True)
q=np.quantile(y,[.001,.01,.05,.25,.5,.75,.95,.99,.999]); labels=['0.1%','1%','5%','25%','50%','75%','95%','99%','99.9%']
ax[0].plot(labels,q*100,marker='o',color=COLORS['train']); ax[0].axhline(0,color='gray',lw=.8); ax[0].set_title('收益率分位数'); ax[0].set_ylabel('收益率（%）'); ax[0].grid(alpha=.25)
ax[1].hist(y*100,bins=160,range=(-15,15),density=True,color=COLORS['train'],alpha=.9); ax[1].axvline(0,color='gray',lw=.8); ax[1].set_title('收益率密度（±15%范围）'); ax[1].set_xlabel('下一日收益率（%）'); ax[1].set_ylabel('密度')
fig.suptitle('训练集标签分布',fontsize=15); fig.savefig(os.path.join(OUT,'eda_returns.png'),bbox_inches='tight'); plt.close(fig)

# 3. Feature distributions
fig, ax = plt.subplots(1,3, figsize=(15,4.8), constrained_layout=True)
for a,tr,te,title,xlab in [(ax[0],tr_amt,te_amt,'成交额','log10(amount + 1)'),(ax[1],tr_vol,te_vol,'成交量','log10(vol + 1)'),(ax[2],tr_close,te_close,'收盘价','log10(close)')]:
    trlog=np.log10(tr+1); telog=np.log10(te+1) if title!='收盘价' else np.log10(te)
    lo,hi=np.quantile(np.concatenate([trlog,telog]),[.001,.999]); bins=np.linspace(lo,hi,70)
    a.hist(trlog,bins=bins,density=True,color=COLORS['train'],alpha=.62,label='训练集'); a.hist(telog,bins=bins,density=True,color=COLORS['test'],alpha=.62,label='测试集'); a.set_title(title+'分布对比'); a.set_xlabel(xlab); a.set_ylabel('密度'); a.legend(frameon=False)
fig.suptitle('主要量价字段：训练集与测试集分布',fontsize=15); fig.savefig(os.path.join(OUT,'eda_feature_shift.png'),bbox_inches='tight'); plt.close(fig)

# 4. Coverage / market state
fig, ax = plt.subplots(2,1, figsize=(15,7), sharex=False, constrained_layout=True)
ax[0].plot(trd.index, trd.n, color=COLORS['train'], label='训练集每日记录数'); ax[0].plot(ted.index, ted.n, color=COLORS['test'], label='测试集每日记录数'); ax[0].set_title('每日股票记录数'); ax[0].set_ylabel('记录数'); ax[0].legend(frameon=False); ax[0].grid(alpha=.2)
ax[1].plot(trd.index, trd.limit_up*100, color=COLORS['red'], label='训练集涨停'); ax[1].plot(trd.index, trd.limit_down*100, color=COLORS['green'], label='训练集跌停'); ax[1].plot(ted.index, ted.limit_up*100, color=COLORS['red'], ls='--', label='测试集涨停'); ax[1].plot(ted.index, ted.limit_down*100, color=COLORS['green'], ls='--', label='测试集跌停'); ax[1].set_title('涨跌停比例随时间变化'); ax[1].set_ylabel('比例（%）'); ax[1].set_xlabel('交易日'); ax[1].legend(ncol=2,frameon=False); ax[1].grid(alpha=.2)
fig.suptitle('样本覆盖与交易状态',fontsize=15); fig.savefig(os.path.join(OUT,'eda_coverage_market_state.png'),bbox_inches='tight'); plt.close(fig)

print('saved', OUT)

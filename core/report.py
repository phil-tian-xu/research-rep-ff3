"""A self-contained research report, figures, and public-data comparisons."""

import base64
import csv
from html import escape
from io import StringIO
import json
import os
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd

from core.data import RAW, ROOT


def read_french_equal_weighted(path):
    """Read the monthly EW block only; French reports percentage returns."""
    with ZipFile(path) as archive:
        filenames = [x for x in archive.namelist() if x.lower().endswith('.csv')]
        if len(filenames) != 1:
            raise ValueError('Expected exactly one CSV in a French archive.')
        text = archive.read(filenames[0]).decode('utf-8-sig')
    lines = text.splitlines()
    starts = [i for i,line in enumerate(lines) if re.search(r'(?:Average\s+)?Equal Weight(?:ed)? Returns\s*--\s*Monthly',line,re.I)]
    if len(starts) != 1:
        raise ValueError(f'Expected one monthly EW block in {path.name}.')
    header, rows = None, []
    for row in csv.reader(lines[starts[0]+1:]):
        row = [value.strip() for value in row]
        if not row or not any(row):
            if rows:
                break
            continue
        if header is None:
            header = row
        elif re.fullmatch(r'\d{6}',row[0]):
            if len(row) != len(header):
                raise ValueError('French monthly row width differs from its header.')
            rows.append(row)
        elif rows:
            break
    if not rows:
        raise ValueError('No monthly equal-weighted observations.')
    frame = pd.DataFrame(rows,columns=['month',*header[1:]])
    frame['month'] = pd.to_datetime(frame.month,format='%Y%m')+pd.offsets.MonthEnd(0)
    if frame.month.duplicated().any():
        raise ValueError('Duplicate French benchmark months.')
    values = frame.set_index('month').apply(pd.to_numeric,errors='raise').replace([-99.99,-999.0],np.nan)
    expected = pd.date_range('1963-07-31','1990-12-31',freq='ME')
    sample = values.reindex(expected)
    if not expected.isin(values.index).all():
        raise ValueError('French benchmark missing calendar months during the original sample.')
    return sample


def write_report(output_dir, results):
    output_dir = Path(output_dir)
    figures = output_dir/'figures'
    figures.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})
    benchmark = json.loads((ROOT/'config/paper_benchmarks.json').read_text())
    blue, orange = '#175b8c', '#d4772b'
    fig, axes = plt.subplots(1,3,figsize=(14,4.2))
    for ax,name,title in zip(axes[:2],['table_ii_size','table_iv_bm'],['Size portfolios (Table II)','Book-to-market portfolios (Table IV)']):
        actual = results['univariate'][name]
        ax.plot(range(len(actual)),benchmark['univariate'][name]['values']['ret'],'o--',color=orange,label='1992 paper',ms=4)
        ax.plot(range(len(actual)),actual.ret,'o-',color=blue,label='Replication',ms=4)
        ax.set_xticks(range(len(actual)),actual.index,rotation=45)
        ax.set_title(title)
        ax.set_ylabel('Average monthly return (%)')
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False)
    matrix = results['matrices']['table_i_returns']
    paper = benchmark['matrices']['table_i_returns']['rows']['All'][1:]
    axes[2].plot(range(1,11),paper,'o--',color=orange,label='1992 paper',ms=4)
    axes[2].plot(range(1,11),matrix.loc['All',range(1,11)],'o-',color=blue,label='Replication',ms=4)
    axes[2].set(title='Beta groups across size deciles (Table I)',xlabel='Pre-ranking beta group',ylabel='Average monthly return (%)')
    axes[2].set_xticks(range(1,11))
    axes[2].grid(axis='y',alpha=.2)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures/'main_results.png')
    plt.close(fig)

    actual = results['matrices']['table_v_returns'].loc[range(1,11),range(1,11)].to_numpy(dtype=float)
    paper_rows = benchmark['matrices']['table_v_returns']['rows']
    paper = np.array([paper_rows[x][1:] for x in ['Small-ME',*[f'ME-{i}' for i in range(2,10)],'Large-ME']])
    fig,axes = plt.subplots(1,3,figsize=(14,4.5))
    low, high = min(actual.min(),paper.min()),max(actual.max(),paper.max())
    difference = actual-paper
    for ax,values,title in zip(axes,[paper,actual,difference],['1992 paper','Replication','Replication minus paper']):
        if ax is axes[2]:
            limit = max(abs(difference).max(),.01)
            im=ax.imshow(values,cmap='RdBu_r',vmin=-limit,vmax=limit)
        else:
            im=ax.imshow(values,cmap='YlGnBu',vmin=low,vmax=high)
        ax.set(title=title,xlabel='Book-to-market group within size',ylabel='Size decile (small to big)')
        ax.set_xticks(range(10),range(1,11))
        ax.set_yticks(range(10),range(1,11))
        fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label='% per month')
    fig.suptitle('Table V: sequential 10 x 10 sorts, equal-weighted returns')
    fig.tight_layout()
    fig.savefig(figures/'table_v_comparison.png')
    plt.close(fig)

    comparison=results['regression_comparison']
    selected=comparison.loc[comparison.model.isin(['beta','size','size_bm'])].copy()
    selected.to_csv(output_dir/'key_results.csv',index=False)
    audit=results['audit']
    size_bm=selected.loc[selected.model=='size_bm'].set_index('variable')
    beta=selected.loc[selected.model=='beta'].iloc[0]
    size_statement = 'negative' if size_bm.loc['ln_me','mean'] < 0 else 'positive'
    bm_statement = 'positive' if size_bm.loc['ln_bm','mean'] > 0 else 'negative'
    conclusion=(f'The joint size/book-to-market regression gives a {size_statement} size coefficient '
        f'({size_bm.loc["ln_me","mean"]:.3f}, t={size_bm.loc["ln_me","t"]:.2f}) and a '
        f'{bm_statement} book-to-market coefficient ({size_bm.loc["ln_bm","mean"]:.3f}, '
        f't={size_bm.loc["ln_bm","t"]:.2f}). The beta-only premium is {beta["mean"]:.3f} '
        f'(t={beta["t"]:.2f}). Coefficients use percentage monthly returns.')
    metrics=[]
    units={'ret':'percentage points/month','beta':'dimensionless','ln_me':'log USD millions',
           'ln_bm':'log ratio','ln_am':'log ratio','ln_ab':'log ratio','ep_dummy':'fraction',
           'ep_positive':'earnings/market equity','firms':'stocks'}
    for path in sorted((output_dir/'main').glob('*_comparison.csv')):
        frame=pd.read_csv(path)
        if 'difference' in frame:
            if 'variable' not in frame:
                frame['variable']='beta' if '_beta' in path.stem else 'ln_me' if '_size' in path.stem else 'ret'
            for variable,group in frame.groupby('variable'):
                values=group.difference.dropna()
                metrics.append({'table':path.stem.replace('_comparison',''),'variable':variable,
                    'unit':units[variable],'compared_cells':len(values),
                    'mean_absolute_difference':values.abs().mean(),'max_absolute_difference':values.abs().max()})
    pd.DataFrame(metrics).to_csv(output_dir/'comparison_metrics.csv',index=False)
    french_html='<p>No public French snapshots are available; the paper remains the primary benchmark.</p>'
    french_dir=output_dir/'french_benchmarks'
    french_dir.mkdir(exist_ok=True)
    french_files=list((RAW/'french').glob('*CSV.zip'))
    for path in french_files:
        if path.name=='F-F_Research_Data_Factors_CSV.zip':
            continue
        frame=read_french_equal_weighted(path)
        frame.to_csv(french_dir/f'{path.stem}_monthly_196307_199012.csv')
        frame.mean().rename('mean_monthly_return_percent').to_csv(french_dir/f'{path.stem}_means.csv')
        if path.name=='100_Portfolios_10x10_CSV.zip':
            if frame.shape[1]!=100:
                raise ValueError('Expected 100 French size/book-to-market portfolios.')
            current=frame.mean().to_numpy().reshape(10,10)
            pd.DataFrame(current,index=range(1,11),columns=range(1,11)).to_csv(french_dir/'french_100_ew_means.csv')
            rows=[]
            for i in range(10):
                for j in range(10):
                    french_series=frame.iloc[:,i*10+j]
                    matched=results['bm_monthly'][(i+1,j+1)].reindex(frame.index).loc[french_series.notna()]
                    rows.append({'size_group':i+1,'bm_group':j+1,'paper_sequential':paper[i,j],
                        'replication_sequential':actual[i,j],'french_independent':current[i,j],
                        'french_months':int(french_series.notna().sum()),
                        'replication_common_months':matched.mean(),
                        'replication_common_months_minus_french':matched.mean()-current[i,j]})
            french_comparison=pd.DataFrame(rows)
            french_comparison.to_csv(french_dir/'size_bm_comparison_different_methods.csv',index=False)
            french_html=('<p>The current French 100-portfolio equal-weighted series is restricted to the same 330 months. '
                'The comparison below is descriptive: French uses independent NYSE size and B/M breakpoints, '
                'whereas this paper uses B/M sorts within size groups and a nonfinancial, accounting-complete sample. '
                'The library also uses a later book-equity definition with preferred-stock adjustments and investment tax credits. '
                'Current French returns use CIZ; the replication uses legacy FIZ. Differences are not a replication error measure. '
                '<a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html">French variable definitions</a>.</p>'
                +f'<p>The French file has {int(frame.isna().sum().sum())} missing portfolio-month observations in this window. '
                'The CSV preserves observation counts and recomputes our averages on each French portfolio’s available months. '
                'Mean absolute difference across 100 cells on these matched months: '
                f'{french_comparison.replication_common_months_minus_french.abs().mean():.3f} percentage points per month.</p>'
                +'<a href="french_benchmarks/size_bm_comparison_different_methods.csv">Download the French comparison</a>')

    def table(frame):
        return frame.to_html(index=False,float_format=lambda value:f'{value:.3f}',border=0,classes='data')
    def embedded_image(name,alt):
        encoded=base64.b64encode((figures/name).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{encoded}" alt="{escape(alt)}">'
    sections=[]
    for name,title in [('table_ii_size','Table II: size'),('table_ii_beta','Table II: pre-ranking beta'),
                       ('table_iv_bm','Table IV: book-to-market'),('table_iv_ep','Table IV: earnings/price')]:
        frame=results['univariate'][name].reset_index(names='portfolio')
        sections.append(f'<details><summary>{title}</summary>{table(frame)}<a href="main/{name}_comparison.csv">Paper comparison CSV</a></details>')
    matrix_sections=[]
    for name,matrix in results['matrices'].items():
        folder='appendix' if 'aii' in name else 'main'
        matrix_sections.append(f'<details><summary>{escape(name.replace("_"," ").title())}</summary>'
            +table(matrix.reset_index(names='size_group'))
            +f'<a href="{folder}/{name}_comparison.csv">Paper comparison CSV</a></details>')
    appendix_html=''
    if results['appendix']:
        for label,summary in results['appendix'].items():
            appendix_html+=f'<h3>Table {label.upper()}: individual-stock regressions</h3>'+table(summary)
        appendix_html+=('<p><a href="appendix/table_ai_properties.csv">Table AI portfolio properties</a> · '
            '<a href="appendix/table_ai_residuals.csv">Table AI residuals</a> · '
            '<a href="appendix/table_aiii_residuals.csv">Table AIII residuals</a> · '
            '<a href="appendix/table_aiv_subperiods.csv">Table AIV subperiod results</a>. '
            'Portfolio regression results and their comparisons are saved in the appendix folder.</p>')
    run_summary={'conclusion':conclusion,'main_months':int(audit['months']),
                 'average_stocks_per_month':audit['mean_stocks_per_month'],
                 'table_v_mae_percent':float(abs(difference).mean()),
                 'main_paper_sample_average_stocks':2267,'tax_variable':audit['tax_variable']}
    (output_dir/'summary.json').write_text(json.dumps(run_summary,indent=2)+'\n')
    tax=audit['tax_variable'].upper()
    nyse_vw=results['nyse_market'].loc['1941':'1990','vwretd']
    nyse_log_mean=np.log1p(nyse_vw).mean()*100
    nyse_arithmetic_mean=nyse_vw.mean()*100
    appendix_description=('Appendix AI–AIV: January 1941–December 1990.' if results['appendix']
                          else 'Appendix omitted for this run.')
    sensitivity_html=''
    sensitivity_path=output_dir/'book_equity_sensitivity.csv'
    if sensitivity_path.is_file():
        sensitivity_html=('<h2>Book-equity sensitivity</h2><p>The alternative CEQ + TXDITC specification '
            'uses deferred taxes plus investment tax credits. It is kept separate from the primary CEQ + TXDB run.</p>'
            +table(pd.read_csv(sensitivity_path))
            +'<p><a href="book_equity_sensitivity.csv">Sensitivity CSV</a> · '
            '<a href="txditc_sensitivity/report.html">Alternative-specification report</a></p>')
    key_table=selected[['model','variable','paper_mean','mean','coefficient_difference','paper_t','t']].rename(
        columns={'mean':'replication_mean','t':'replication_t'})
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fama and French (1992) replication</title><style>
body{{font:16px/1.55 system-ui,sans-serif;color:#172b3a;background:#f3f5f7;margin:0}}main{{max-width:1180px;margin:35px auto;background:white;padding:40px;border-radius:8px}}
h1{{font-size:32px;line-height:1.15}}h2{{margin-top:38px;border-bottom:1px solid #d9e2e9;padding-bottom:8px}}h3{{font-size:19px}}
.muted{{color:#5e6e78}}.callout{{background:#edf5fb;border-left:4px solid #175b8c;padding:18px 22px}}img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto;margin:14px 0 24px;white-space:nowrap}}th,td{{padding:8px 10px;border-bottom:1px solid #dee5eb;text-align:right}}th{{background:#edf2f6}}a{{color:#175b8c}}
details{{border:1px solid #dee5eb;border-radius:4px;margin:12px 0;padding:12px}}summary{{cursor:pointer;font-weight:600}}code{{background:#edf2f6;padding:2px 5px}}li{{margin:8px 0}}
</style><main><p class="muted">Empirical asset pricing · original sample · modern data vintage</p>
<h1>Fama and French (1992)<br>The Cross-Section of Expected Stock Returns</h1>
<p>Main Tables I–VI: July 1963–December 1990. {appendix_description}</p>
<p>Book-equity specification: <strong>CEQ + {tax}</strong>.</p>
<div class="callout">{escape(conclusion)}</div>
<p>The table below distinguishes univariate size from size conditional on book-to-market. A negative coefficient alone does not establish statistical significance; use its reported t-statistic.</p>
<h2>Key results against the paper</h2>
<p>The main sample contains <strong>{audit['monthly_stock_observations']:,} stock-months</strong> over <strong>{audit['months']} months</strong>, averaging <strong>{audit['mean_stocks_per_month']:,.1f} stocks/month</strong>. The paper reports 2,267 stocks/month. All reported results below are computed from the stock-level inputs; published values are separate benchmarks.</p>
{table(key_table)}
<p><a href="key_results.csv">Key results CSV</a> · <a href="main/table_iii_comparison.csv">Full Table III comparison</a> · <a href="main/monthly_fm_slopes.csv">Monthly regression slopes</a></p>
{embedded_image('main_results.png','Portfolio average returns compared with the original paper')}
<h2>Size and book-to-market: Table V</h2>
{embedded_image('table_v_comparison.png','Paper and replication Table V return matrices and differences')}
<p>Across the 100 interior cells, mean absolute deviation from the paper is <strong>{abs(difference).mean():.3f} percentage points/month</strong>. This is a descriptive comparison; original numbers are rounded and source databases have since changed.</p>
<h2>All portfolio tables and regression specifications</h2>
{''.join(matrix_sections)}{''.join(sections)}
<h3>Table III: all reported specifications</h3>{table(comparison[['model','variable','paper_mean','mean','coefficient_difference','paper_t','t']])}
<h3>Table VI: subperiod statistics</h3><p>NYSE index rows use log returns, which reproduce the paper’s reported mean/std/t. This reporting convention is inferred from those numbers; arithmetic versions are saved separately. Stock-regression coefficients retain percentage simple-return units.</p>{table(results['subperiods'])}
<p><a href="main/table_vi_subperiods.csv">Table VI CSV</a> · <a href="main/sample_counts.csv">Monthly sample counts</a> · <a href="comparison_metrics.csv">Comparison metrics</a></p>
<h2>Appendix: beta and size, NYSE 1941–1990</h2>{appendix_html or '<p>Appendix was not requested for this run.</p>'}
<h2>Comparison with Kenneth French’s website</h2>{french_html}
{sensitivity_html}<h2>Methodology and interpretation</h2><ul>
<li>Main formation sample: NYSE/AMEX/Nasdaq common shares (CRSP share codes 10/11), positive market equity, nonfinancial firms (exclude SIC 6000–6999), positive book equity/assets, available earnings, and 24–60 prior monthly return observations. Historical Compustat SIC is used when available, otherwise dated CRSP SIC.</li>
<li>Fiscal year-ends in calendar year t−1 explain returns from July t through June t+1. Size is June market equity; valuation denominators use December t−1 market equity. Both market equity and accounts are in USD millions.</li>
<li>Book equity = CEQ + {tax}. The primary TXDB specification follows the paper’s common-equity-plus-balance-sheet-deferred-taxes definition. Earnings = IB + TXDI − DVP. Missing tax/dividend add-ons are zero; missing base fields remain missing. Latest fiscal year-end per calendar year is used; domestic USD Compustat records only.</li>
<li>CCM LC/LU/LS links with primary status P/C must be valid at formation. Overlaps prefer P, then LC/LU/LS and the latest link start. No extra two-year Compustat seasoning rule is imposed.</li>
<li>NYSE size cutoffs use all NYSE common stocks with size data, including financial firms. Size–beta sorts use NYSE beta breakpoints within size; size–B/M sorts use all eligible stocks within size. One-dimensional B/M and E/P cutoffs use the full eligible sample, as in the paper.</li>
<li>Pre-ranking beta is the sum of current and one-lag market slopes in a raw-return regression with an intercept. The lag is calendar-aligned before taking the five-year window. Post-ranking betas use full-sample portfolio returns and are assigned to constituent stocks. This intentional ex-post estimation matches the paper; these are not out-of-sample trading signals.</li>
<li>Portfolio returns are equal-weighted across observed surviving constituents each month. No future-return availability is required at formation and missing returns are not filled. The legacy CRSP RET series is used without an additional DLRET adjustment; this assumption may affect differences from the original vintage.</li>
<li>Monthly cross-sectional OLS uses percentage stock returns and an intercept. Table III ratios are clipped at monthly 0.5/99.5 percentiles before logs. FM t-statistics use the conventional time-series standard error of monthly slopes; no Newey–West adjustment is added.</li>
<li>The appendix forms portfolios at December year-end using NYSE stocks and the CRSP NYSE index (MSIA). Financial stocks are included in the appendix, whose tests do not require accounts.</li>
<li>Only the descriptive NYSE market rows in Tables VI/AIV use 100 log(1+R). This reporting convention is inferred from the published means/std/t; 1941–1990 NYSE VW has a log mean of {nyse_log_mean:.3f}% versus the paper’s 0.93%, while the arithmetic mean is {nyse_arithmetic_mean:.3f}%. Separate arithmetic diagnostic CSVs are retained. Market betas and all individual-stock regressions still use raw simple returns.</li>
</ul><h2>Reproducibility and data provenance</h2>
<p>Run <code>uv sync</code>, then <code>uv run python main.py --skip-download</code> from the project. Run tests with <code>uv run pytest</code>. Fresh downloads use WRDS_USERNAME and the existing PostgreSQL password file; credentials are never saved in the project.</p>
<p><a href="data_audit.json">Data audit</a> · <a href="run_manifest.json">Source SQL, checksums, and run manifest</a> · <a href="summary.json">Machine-readable summary</a>. Raw and intermediate data are separate from final outputs. The momentum project was used only as a read-only data/reference source.</p>
<p><a href="main/formation_attrition.csv">Annual sample attrition by filter</a> · <a href="main/accounting_coverage.csv">Annual accounting-field availability</a>. These show how source coverage and sample screens affect the historical cohorts.</p>
<p class="muted">Sources: Fama and French (1992), Journal of Finance 47(2), 427–465; CRSP/Compustat via WRDS; <a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html">Kenneth French Data Library</a>; <a href="https://www.crsp.org/crsp_pdf/crsp-historical-indexes-guide/">CRSP Historical Indexes Guide</a>. Modern-vintage reconstruction with explicit implementation choices; numerical identity with the 1992 vintage is not assumed.</p></main></html>'''
    (output_dir/'report.html').write_text(html,encoding='utf-8')

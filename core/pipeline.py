"""Run main Tables I--VI and appendix Tables AI--AIV from raw inputs."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from core.analysis import (CHARACTERISTICS, MODELS, SPLIT_LABELS, estimate_beta,
    fama_macbeth, form_appendix_sample, form_main_sample, holding_panel,
    portfolio_matrix, portfolio_summary, post_ranking_betas, pre_ranking_betas,
    prepare_accounts, prepare_crsp, summarize_slopes)
from core.data import RAW, ROOT


def compare_matrix(actual, benchmark):
    labels = ['All', 'Small-ME', *[f'ME-{x}' for x in range(2, 10)], 'Large-ME']
    expected = pd.DataFrame.from_dict(benchmark['rows'], orient='index', columns=['All', *range(1, 11)])
    expected = expected.reindex(labels)
    expected.index = ['All', *range(1, 11)]
    records = []
    for row in actual.index:
        for column in actual.columns:
            paper = expected.loc[row, column]
            records.append({'size_group': row, 'second_group': column, 'replication': actual.loc[row, column],
                            'paper': paper, 'difference': actual.loc[row, column] - paper})
    return pd.DataFrame(records)


def compare_univariate(actual, benchmark):
    expected = pd.DataFrame(benchmark['values'], index=actual.index)
    records = []
    for group in actual.index:
        for variable in expected:
            records.append({'portfolio': group, 'variable': variable, 'replication': actual.loc[group, variable],
                            'paper': expected.loc[group, variable],
                            'difference': actual.loc[group, variable] - expected.loc[group, variable]})
    return pd.DataFrame(records)


def subperiod_summary(slopes, market, periods, market_log=True):
    rows = []
    for label, start, end in periods:
        subset = slopes.loc[slopes.month.between(start, end)]
        summary = summarize_slopes(subset)
        summary.insert(0, 'period', label)
        summary['scale']='FM slope (percent stock returns)'
        rows.append(summary)
        for name in ['vwretd', 'ewretd']:
            source = market.loc[start:end, name].dropna()
            # Log-return means/std/t reproduce the paper's NYSE index rows.
            # This reporting convention is inferred from the published numbers;
            # it does NOT alter market inputs to beta estimation or stock OLS.
            series = (np.log1p(source) if market_log else source) * 100
            std = series.std(ddof=1)
            mean = series.mean()
            rows.append(pd.DataFrame([{'period': label, 'model': 'NYSE market', 'variable': name,
                'mean': mean, 'std': std, 'count': len(series), 'se': std/np.sqrt(len(series)),
                't': mean / (std/np.sqrt(len(series))),
                'scale':'log index return (%)' if market_log else 'arithmetic index return (%)'}]))
    return pd.concat(rows, ignore_index=True)


def compare_regressions(summary, benchmark, table, unit=None):
    rows = [r for r in benchmark['regressions'] if r['table'] == table and r.get('unit') == unit]
    comparison = pd.DataFrame(rows).merge(summary, on=['model', 'variable'], validate='1:1')
    comparison['coefficient_difference'] = comparison['mean'] - comparison['paper_mean']
    comparison['t_difference'] = comparison['t'] - comparison['paper_t']
    if 'paper_se' in comparison:
        comparison['se_difference'] = comparison['se'] - comparison['paper_se']
    return comparison


def compare_subperiods(summary, benchmark, table):
    expected = pd.DataFrame([r for r in benchmark['subperiods'] if r['table']==table])
    comparison = expected.merge(summary,on=['period','model','variable'],validate='1:1')
    comparison['mean_difference'] = comparison['mean']-comparison['paper_mean']
    comparison['t_difference'] = comparison['t']-comparison['paper_t']
    comparison['std_difference'] = comparison['std']-comparison['paper_std']
    return comparison


def run_replication(output_dir, include_appendix=True, tax_variable='txdb'):
    output_dir = Path(output_dir)
    main_dir = output_dir / 'main'
    main_dir.mkdir(parents=True, exist_ok=True)
    intermediate = ROOT / 'data' / 'intermediate' / tax_variable
    intermediate.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads((ROOT/'config/paper_benchmarks.json').read_text())
    print('Loading raw CRSP and accounting extracts...', flush=True)
    crsp = prepare_crsp(pd.read_parquet(RAW/'crsp_monthly.parquet'))
    accounts = prepare_accounts(pd.read_parquet(RAW/'compustat_annual.parquet'), tax_variable)
    links = pd.read_parquet(RAW/'ccm_links.parquet')
    markets = {}
    for name in ['market_returns', 'nyse_market']:
        frame = pd.read_parquet(RAW/f'{name}.parquet')
        frame['month'] = frame.date.dt.to_period('M').dt.to_timestamp('M')
        markets[name] = frame.set_index('month')[['vwretd','ewretd']].sort_index()
    print('Main sample: estimating five-year pre-ranking betas, 1963--1990...', flush=True)
    pre = pre_ranking_betas(crsp, markets['market_returns'].vwretd, 6, 1963, 1990)
    pre.to_parquet(intermediate/'main_pre_betas.parquet', index=False)
    formation, audit = form_main_sample(crsp, accounts, links, pre)
    pd.DataFrame(audit['formation_attrition_by_year']).to_csv(main_dir/'formation_attrition.csv',index=False)
    accounts.groupby('account_year').agg(firms=('gvkey','size'),ceq_available=('ceq','count'),
        assets_available=('at','count'),earnings_available=('ib','count')).to_csv(main_dir/'accounting_coverage.csv')
    formation.to_parquet(intermediate/'main_formation.parquet', index=False)
    panel = holding_panel(crsp, formation)
    betas, portfolio_returns = post_ranking_betas(panel, markets['market_returns'].vwretd)
    if not portfolio_returns.index.equals(pd.date_range('1963-07-31','1990-12-31',freq='ME')):
        raise ValueError('Main sample does not cover all 330 months.')
    panel = panel.merge(betas[['size_beta','beta']], on='size_beta', validate='m:1')
    betas.to_csv(main_dir/'post_ranking_betas.csv', index=False)
    portfolio_returns.to_csv(main_dir/'size_beta_monthly_returns.csv')
    bm_monthly=panel.groupby(['month','size10','bm10'],observed=True).ret.mean().unstack(['size10','bm10'])*100
    bm_monthly.to_csv(main_dir/'size_bm_monthly_returns_percent.csv')
    panel.to_parquet(intermediate/'main_stock_months.parquet', index=False)
    audit.update({'tax_variable': tax_variable, 'monthly_stock_observations': len(panel),
        'months': panel.month.nunique(), 'mean_stocks_per_month': len(panel)/panel.month.nunique(),
        'min_stocks_per_month': int(panel.groupby('month').size().min()),
        'max_stocks_per_month': int(panel.groupby('month').size().max()),
        'missing_size_beta_portfolio_months': int(portfolio_returns.isna().sum().sum()),
        'minimum_post_ranking_beta_months': int(betas.months.min()),
        'raw_crsp_rows': len(crsp), 'annual_accounting_rows_after_selection': len(accounts)})
    panel.groupby('month').size().rename('stocks').to_csv(main_dir/'sample_counts.csv')
    formation.groupby('formation_year').size().rename('stocks').to_csv(main_dir/'formation_counts.csv')
    print(f'Main sample: {len(panel):,} stock-months, {audit["mean_stocks_per_month"]:.1f} stocks/month.', flush=True)

    matrices = {}
    for name, second, variable, market in [
        ('table_i_returns','beta10','ret',None),
        ('table_i_beta','beta10','ret',markets['market_returns'].vwretd),
        ('table_i_size','beta10','ln_me',None),
        ('table_v_returns','bm10','ret',None),
    ]:
        actual = portfolio_matrix(panel, second, variable, market)
        actual.to_csv(main_dir/f'{name}.csv')
        compare_matrix(actual, benchmark['matrices'][name]).to_csv(main_dir/f'{name}_comparison.csv', index=False)
        matrices[name] = actual
    univariate = {}
    for name, group in [('table_ii_size','size12'),('table_ii_beta','beta12'),
                         ('table_iv_bm','bm12'),('table_iv_ep','ep13')]:
        actual = portfolio_summary(panel, group)
        actual.index = (['0'] if group == 'ep13' else []) + SPLIT_LABELS
        actual.to_csv(main_dir/f'{name}.csv')
        compare_univariate(actual, benchmark['univariate'][name]).to_csv(main_dir/f'{name}_comparison.csv', index=False)
        univariate[name] = actual
    print('Main sample: running monthly Fama--MacBeth regressions...', flush=True)
    slopes, _ = fama_macbeth(panel)
    slopes.to_csv(main_dir/'monthly_fm_slopes.csv', index=False)
    summary = summarize_slopes(slopes)
    summary.to_csv(main_dir/'table_iii_regressions.csv', index=False)
    regression_comparison = compare_regressions(summary, benchmark, 'III')
    regression_comparison.to_csv(main_dir/'table_iii_comparison.csv', index=False)
    main_periods = [
        ('1963-1990','1963-07-31','1990-12-31'), ('1963-1976','1963-07-31','1976-12-31'),
        ('1977-1990','1977-01-31','1990-12-31')]
    vi_slopes=slopes.loc[slopes.model.isin(['size_bm','beta_size_bm'])]
    vi = subperiod_summary(vi_slopes, markets['nyse_market'], main_periods)
    vi.to_csv(main_dir/'table_vi_subperiods.csv', index=False)
    compare_subperiods(vi,benchmark,'VI').to_csv(main_dir/'table_vi_comparison.csv',index=False)
    subperiod_summary(vi_slopes,markets['nyse_market'],main_periods,False).to_csv(main_dir/'table_vi_arithmetic_market_diagnostic.csv',index=False)
    seasonal = slopes.assign(season=np.where(slopes.month.dt.month == 1,'January','February-December'))
    seasonal.groupby(['season','model','variable']).coefficient.agg(['mean','std','count']).to_csv(main_dir/'seasonality.csv')

    appendix_results = {}
    if include_appendix:
        directory = output_dir/'appendix'
        directory.mkdir(exist_ok=True)
        print('Appendix: estimating December pre-ranking betas, 1940--1989...', flush=True)
        pre_appendix = pre_ranking_betas(crsp, markets['nyse_market'].vwretd, 12, 1940, 1989)
        pre_appendix.to_parquet(intermediate/'appendix_pre_betas.parquet', index=False)
        appendix_formation = form_appendix_sample(crsp, pre_appendix)
        appendix_formation.to_parquet(intermediate/'appendix_formation.parquet', index=False)
        base = holding_panel(crsp, appendix_formation, 12, '1941-01-31','1990-12-31')
        if base.month.nunique() != 600:
            raise ValueError('Appendix does not contain all 600 calendar months.')
        basic_models = {key:MODELS[key] for key in ['beta','size','beta_size']}
        for label, grouping, paper_table in [('ai','size12','AI'),('aiii','size_beta','AIII')]:
            print(f'Appendix Table {paper_table}: stock and portfolio regressions...', flush=True)
            estimates, returns = post_ranking_betas(base, markets['nyse_market'].vwretd, grouping)
            estimates.to_csv(directory/f'table_{label}_post_ranking_betas.csv',index=False)
            returns.to_csv(directory/f'table_{label}_monthly_portfolio_returns.csv')
            working = base.merge(estimates[[grouping,'beta']], on=grouping, validate='m:1')
            stock_slopes, residuals = fama_macbeth(working, basic_models, False, ['size12','beta12'])
            stock_summary = summarize_slopes(stock_slopes)
            stock_slopes.to_csv(directory/f'table_{label}_monthly_stock_slopes.csv', index=False)
            stock_summary.to_csv(directory/f'table_{label}_stock_regressions.csv', index=False)
            compare_regressions(stock_summary, benchmark, paper_table, 'stock').to_csv(
                directory/f'table_{label}_stock_comparison.csv', index=False)
            portfolio_panel = working.groupby(['month', grouping], observed=True)[['ret','beta','ln_me']].mean().reset_index()
            portfolio_slopes, _ = fama_macbeth(portfolio_panel, basic_models, False)
            portfolio_slopes.to_csv(directory/f'table_{label}_monthly_portfolio_slopes.csv', index=False)
            portfolio_summary_fm = summarize_slopes(portfolio_slopes)
            portfolio_summary_fm.to_csv(directory/f'table_{label}_portfolio_regressions.csv', index=False)
            compare_regressions(portfolio_summary_fm, benchmark, paper_table, 'portfolio').to_csv(
                directory/f'table_{label}_portfolio_comparison.csv', index=False)
            residuals.to_csv(directory/f'table_{label}_monthly_residuals.csv', index=False)
            residual_summary = residuals.groupby(['model','grouping','portfolio']).residual.agg(['mean','std','count'])
            residual_summary['se'] = residual_summary['std']/np.sqrt(residual_summary['count'])
            residual_summary.to_csv(directory/f'table_{label}_residuals.csv')
            expected_residuals=pd.DataFrame([r for r in benchmark['residuals'] if r['table']==paper_table])
            residual_comparison=expected_residuals.merge(residual_summary.reset_index(),on=['model','grouping','portfolio'],validate='1:1')
            residual_comparison['mean_difference']=residual_comparison['mean']-residual_comparison['paper_mean']
            residual_comparison['se_difference']=residual_comparison['se']-residual_comparison['paper_se']
            residual_comparison.to_csv(directory/f'table_{label}_residuals_comparison.csv',index=False)
            if label == 'ai':
                properties = portfolio_summary(working,'size12')
                properties = properties.join(estimates.set_index(grouping)[['beta_se']])
                for portfolio in returns:
                    d = pd.concat([returns[portfolio].rename('ret'), markets['nyse_market'].vwretd.rename('market')], axis=1).dropna()
                    x = np.column_stack([np.ones(len(d)), d.market])
                    coef = np.linalg.lstsq(x, d.ret, rcond=None)[0]
                    residual = d.ret.to_numpy()-x@coef
                    se = np.sqrt((residual@residual)/(len(d)-2)*np.linalg.inv(x.T@x)[1,1])
                    properties.loc[portfolio,'simple_beta'] = coef[1]
                    properties.loc[portfolio,'simple_beta_se'] = se
                properties.index = SPLIT_LABELS
                properties.to_csv(directory/'table_ai_properties.csv')
                compare_univariate(properties, {'values':benchmark['table_ai']}).to_csv(directory/'table_ai_properties_comparison.csv', index=False)
            else:
                for name, variable, market in [('table_aii_returns','ret',None),
                    ('table_aii_beta','ret',markets['nyse_market'].vwretd),('table_aii_size','ln_me',None)]:
                    matrix = portfolio_matrix(working,'beta10',variable,market)
                    matrix.to_csv(directory/f'{name}.csv')
                    compare_matrix(matrix,benchmark['matrices'][name]).to_csv(directory/f'{name}_comparison.csv', index=False)
                    matrices[name] = matrix
                periods = [('1941-1990','1941-01-31','1990-12-31'),('1941-1965','1941-01-31','1965-12-31'),
                           ('1966-1990','1966-01-31','1990-12-31')]
                periods += [(f'{year}-{year+9}',f'{year}-01-31',f'{year+9}-12-31') for year in range(1941,1990,10)]
                aiv=subperiod_summary(stock_slopes,markets['nyse_market'],periods)
                aiv.to_csv(directory/'table_aiv_subperiods.csv',index=False)
                compare_subperiods(aiv,benchmark,'AIV').to_csv(directory/'table_aiv_comparison.csv',index=False)
                subperiod_summary(stock_slopes,markets['nyse_market'],periods,False).to_csv(directory/'table_aiv_arithmetic_market_diagnostic.csv',index=False)
                working.to_parquet(intermediate/'appendix_stock_months.parquet', index=False)
            appendix_results[label] = stock_summary
        audit.update({'appendix_months': base.month.nunique(), 'appendix_stock_months': len(base),
                       'appendix_average_stocks': len(base)/600})

    (output_dir/'data_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (output_dir/'run_manifest.json').write_text(json.dumps({
        'target': 'Fama and French (1992), main Tables I-VI and appendix AI-AIV',
        'tax_variable': tax_variable, 'include_appendix': include_appendix,
        'source_metadata': {p.stem:json.loads(p.read_text()) for p in RAW.glob('*.json')},
        'french_metadata':{p.stem:json.loads(p.read_text()) for p in (RAW/'french').glob('*.json')},
        'paper_benchmark_file': 'config/paper_benchmarks.json',
        'paper_benchmark_sha256':sha256((ROOT/'config/paper_benchmarks.json').read_bytes()).hexdigest(),
        'run_time_utc':datetime.now(timezone.utc).isoformat(),
        'python_version':sys.version,
        'packages':{name:version(name) for name in ['numpy','pandas','pyarrow','matplotlib','sqlalchemy','psycopg2-binary']},
    }, indent=2)+'\n')
    return {'audit':audit, 'regression_comparison':regression_comparison, 'univariate':univariate,
            'matrices':matrices, 'slopes':slopes, 'subperiods':vi, 'appendix':appendix_results,
            'bm_monthly':bm_monthly, 'nyse_market':markets['nyse_market']}

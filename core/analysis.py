"""Paper-defined annual sorts, Dimson betas, and Fama--MacBeth regressions.

Returns are decimals in source/holding data and percent in reported regressions.
Full-sample post-ranking portfolio betas are intentional: this is the paper's
ex-post asset-pricing test, not a tradable forecasting exercise.
"""

import numpy as np
import pandas as pd


DECILES = np.arange(.1, 1, .1)
SPLIT_DECILES = np.array([.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
SPLIT_LABELS = ['1A', '1B', '2', '3', '4', '5', '6', '7', '8', '9', '10A', '10B']
CHARACTERISTICS = ['ln_me', 'ln_bm', 'ln_am', 'ln_ab', 'ep_dummy', 'ep_positive']
MODELS = {
    'beta': ['beta'], 'size': ['ln_me'], 'beta_size': ['beta', 'ln_me'],
    'bm': ['ln_bm'], 'leverage': ['ln_am', 'ln_ab'],
    'ep': ['ep_dummy', 'ep_positive'], 'size_bm': ['ln_me', 'ln_bm'],
    'size_leverage': ['ln_me', 'ln_am', 'ln_ab'],
    'size_ep': ['ln_me', 'ep_dummy', 'ep_positive'],
    'size_bm_ep': ['ln_me', 'ln_bm', 'ep_dummy', 'ep_positive'],
    'size_leverage_ep': ['ln_me', 'ln_am', 'ln_ab', 'ep_dummy', 'ep_positive'],
    'beta_size_bm': ['beta', 'ln_me', 'ln_bm'],
}


def assign_groups(values, reference, quantiles=DECILES):
    """NYSE/reference empirical quantiles; equal breakpoint values stay together."""
    reference = np.asarray(reference.dropna(), dtype=float)
    if len(reference) < len(quantiles) + 1:
        raise ValueError('Insufficient reference stocks to form the requested portfolios.')
    cuts = np.quantile(reference, quantiles)
    result = pd.Series(np.searchsorted(cuts, values.to_numpy(dtype=float), side='left') + 1,
                       index=values.index, dtype='Int64')
    result.loc[values.isna()] = pd.NA
    return result


def estimate_beta(returns, market, min_observations=24):
    """Intercept + current/one-lag RAW market returns, calendar-aligned."""
    data = pd.concat([returns.rename('ret'), market.rename('market'),
                      market.shift(1).rename('market_lag')], axis=1).dropna()
    if len(data) < min_observations:
        return (np.nan, np.nan, len(data))
    x = np.column_stack([np.ones(len(data)), data[['market', 'market_lag']].to_numpy()])
    if np.linalg.matrix_rank(x) < 3:
        return (np.nan, np.nan, len(data))
    y = data['ret'].to_numpy()
    slopes = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ slopes
    covariance = np.linalg.inv(x.T @ x) * (residual @ residual) / (len(y) - 3)
    beta_se = np.sqrt(covariance[1, 1] + covariance[2, 2] + 2 * covariance[1, 2])
    return (slopes[1:].sum(), beta_se, len(data))


def pre_ranking_betas(crsp, market, formation_month, start_year, end_year):
    """Batched normal equations on exactly 60 calendar months, minimum 24 pairs.

    Market lag is formed BEFORE slicing each five-year window. A gap in a
    stock's history never changes the calendar meaning of the market lag.
    """
    returns = crsp.pivot(index='month', columns='permno', values='ret')
    output = []
    market_lag = market.shift(1)
    for year in range(start_year, end_year + 1):
        end = pd.Timestamp(year, formation_month, 1) + pd.offsets.MonthEnd(0)
        months = pd.date_range(end=end, periods=60, freq='ME')
        y = returns.reindex(months).to_numpy(dtype=float)
        x = np.column_stack([np.ones(60), market.reindex(months), market_lag.reindex(months)])
        valid = np.isfinite(y) & np.isfinite(x).all(axis=1)[:, None]
        weights = valid.astype(float)
        count = valid.sum(axis=0)
        x = np.nan_to_num(x)
        xtx = np.einsum('ti,tj,tn->nij', x, x, weights, optimize=True)
        xty = x.T @ np.where(valid, y, 0)
        usable = (count >= 24) & (np.linalg.det(xtx) > 1e-12)
        slopes = np.linalg.solve(xtx[usable], xty[:, usable].T[..., None])[..., 0]
        output.append(pd.DataFrame({'permno': returns.columns[usable], 'formation_year': year,
                                    'pre_beta': slopes[:, 1:].sum(axis=1), 'beta_months': count[usable]}))
    return pd.concat(output, ignore_index=True)


def prepare_crsp(raw):
    frame = raw.copy()
    if frame.duplicated(['permno', 'date']).any():
        raise ValueError('Duplicate CRSP permno/date keys; investigate name-history overlaps.')
    frame['month'] = frame['date'].dt.to_period('M').dt.to_timestamp('M')
    frame['me'] = frame['prc'].abs() * frame['shrout'] / 1000
    if ((frame['ret'] < -1) & frame['ret'].notna()).any():
        raise ValueError('CRSP contains invalid numeric return codes below -1.')
    return frame.sort_values(['permno', 'month'])


def prepare_accounts(raw, tax_variable='txdb'):
    """Common equity + balance-sheet deferred taxes; E=IB+TXDI-DVP.

    Keep USD domestic population and the latest fiscal year-end in calendar
    t-1. Missing tax/dividend add-ons are explicitly treated as zero, while
    missing CEQ, AT, and IB remain missing. This assumption is audited.
    """
    frame = raw.loc[(raw['popsrc'] == 'D') & (raw['curcd'] == 'USD')].copy()
    frame['account_year'] = frame['datadate'].dt.year
    if frame.duplicated(['gvkey', 'datadate']).any():
        raise ValueError('Duplicate domestic Compustat firm/fiscal-date records.')
    frame = frame.sort_values(['gvkey', 'datadate']).drop_duplicates(['gvkey', 'account_year'], keep='last')
    frame['be'] = frame['ceq'] + frame[tax_variable].fillna(0)
    frame['earnings'] = frame['ib'] + frame['txdi'].fillna(0) - frame['dvp'].fillna(0)
    frame['formation_year'] = frame['account_year'] + 1
    return frame


def form_main_sample(crsp, accounts, links, pre_betas):
    june = crsp.loc[(crsp['month'].dt.month == 6) & crsp['month'].dt.year.between(1963, 1990)].copy()
    june['formation_year'] = june['month'].dt.year
    # Size breakpoints use ALL eligible NYSE common stocks, including firms
    # without Compustat and financials; this differs from the beta cutoffs.
    nyse_size = june.loc[(june.exchcd == 1) & june.shrcd.isin([10, 11]) & (june.me > 0)]
    december = crsp.loc[crsp['month'].dt.month == 12, ['permno', 'month', 'me']].copy()
    december['formation_year'] = december['month'].dt.year + 1
    december = december.rename(columns={'me': 'dec_me'}).drop(columns='month')
    eligible = june.loc[june.shrcd.isin([10, 11]) & june.exchcd.isin([1, 2, 3]) & (june.me > 0)]
    eligible = eligible.merge(december, on=['permno', 'formation_year'], validate='1:1')
    eligible = eligible.merge(pre_betas, on=['permno', 'formation_year'], validate='1:1')
    active_links = links.loc[links.linkprim.isin(['P', 'C'])].copy()
    linked = eligible.merge(active_links, on='permno')
    linked = linked.loc[(linked.linkdt.isna() | (linked.month >= linked.linkdt))
                        & (linked.linkenddt.isna() | (linked.month <= linked.linkenddt))]
    # Resolve the issuer identity BEFORE looking at accounting availability,
    # earnings, or book equity. Otherwise a secondary link could be chosen
    # simply because its accounts pass our economic sample filters.
    overlapping_link_rows = int(linked.duplicated(['permno','formation_year']).sum())
    linked['primary_rank'] = linked.linkprim.map({'P': 0, 'C': 1})
    linked['link_rank'] = linked.linktype.map({'LC': 0, 'LU': 1, 'LS': 2})
    linked = linked.sort_values(['primary_rank','link_rank','linkdt','gvkey'],ascending=[True,True,False,True])
    linked = linked.drop_duplicates(['permno','formation_year'])
    linked = linked.merge(accounts, on=['gvkey', 'formation_year'])
    linked['sic'] = linked['sich'].fillna(linked['siccd'])
    audit = {'june_common_with_december_and_beta': len(eligible), 'linked_before_filters': len(linked),
             'missing_sic': int(linked.sic.isna().sum()),
             'financial': int(linked.sic.between(6000, 6999).sum()),
             'negative_or_zero_be': int((linked.be <= 0).sum()),
             'missing_txdb': int(linked.txdb.isna().sum()), 'missing_txdi': int(linked.txdi.isna().sum()),
             'missing_dvp': int(linked.dvp.isna().sum()),
             'overlapping_link_rows': overlapping_link_rows,
             'filter_count_stage':'Linked formation rows, before economic/accounting filters'}
    nonfinancial = linked.sic.notna() & ~linked.sic.between(6000, 6999)
    positive_book = nonfinancial & (linked.be > 0)
    positive_assets = positive_book & (linked['at'] > 0)
    positive_december_me = positive_assets & (linked.dec_me > 0)
    final_valid = positive_december_me & linked.earnings.notna()
    attrition = pd.concat({
        'crsp_with_december_and_beta':eligible.groupby('formation_year').size(),
        'linked_accounts':linked.groupby('formation_year').size(),
        'known_nonfinancial':linked.loc[nonfinancial].groupby('formation_year').size(),
        'positive_book_equity':linked.loc[positive_book].groupby('formation_year').size(),
        'positive_assets':linked.loc[positive_assets].groupby('formation_year').size(),
        'positive_december_me':linked.loc[positive_december_me].groupby('formation_year').size(),
        'available_earnings':linked.loc[final_valid].groupby('formation_year').size(),
    },axis=1).fillna(0).astype(int)
    audit['formation_attrition_by_year']=attrition.reset_index().to_dict(orient='records')
    linked = linked.loc[final_valid].copy()
    linked['ln_me'] = np.log(linked.me.astype(float))
    linked['bm'] = linked.be / linked.dec_me
    linked['am'] = linked['at'] / linked.dec_me
    linked['ab'] = linked['at'] / linked.be
    linked['ep'] = linked.earnings / linked.dec_me
    linked['ep_dummy'] = (linked.earnings < 0).astype(float)
    linked['ep_positive'] = linked.ep.clip(lower=0)
    for ratio in ['bm', 'am', 'ab']:
        linked['ln_' + ratio] = np.log(linked[ratio].astype(float))
    groups = []
    for year, sample in linked.groupby('formation_year', sort=True):
        sample = sample.copy()
        sample['size10'] = assign_groups(sample.me, nyse_size.loc[nyse_size.formation_year == year, 'me'])
        sample['size12'] = assign_groups(sample.me, nyse_size.loc[nyse_size.formation_year == year, 'me'], SPLIT_DECILES)
        sample['beta12'] = assign_groups(sample.pre_beta, sample.loc[sample.exchcd == 1, 'pre_beta'], SPLIT_DECILES)
        sample['bm12'] = assign_groups(sample.bm, sample.bm, SPLIT_DECILES)
        positive = sample.ep >= 0
        sample['ep13'] = 0
        sample.loc[positive, 'ep13'] = assign_groups(sample.loc[positive, 'ep'], sample.loc[positive, 'ep'], SPLIT_DECILES)
        for _, size_group in sample.groupby('size10'):
            sample.loc[size_group.index, 'beta10'] = assign_groups(
                size_group.pre_beta, size_group.loc[size_group.exchcd == 1, 'pre_beta'])
            sample.loc[size_group.index, 'bm10'] = assign_groups(size_group.bm, size_group.bm)
        sample['size_beta'] = (sample.size10 - 1) * 10 + sample.beta10.astype(int)
        groups.append(sample)
    formation = pd.concat(groups, ignore_index=True)
    audit['final_formation_rows'] = len(formation)
    return formation, audit


def holding_panel(crsp, formation, formation_month=6, start='1963-07-31', end='1990-12-31'):
    returns = crsp.loc[crsp.month.between(start, end), ['permno', 'month', 'ret']].copy()
    returns['formation_year'] = returns.month.dt.year - (returns.month.dt.month <= formation_month)
    if formation_month == 12:
        returns['formation_year'] = returns.month.dt.year - 1
    needed = ['permno', 'formation_year', 'me', 'ln_me', 'bm', 'am', 'ab', 'ep',
              'ln_bm', 'ln_am', 'ln_ab', 'ep_dummy', 'ep_positive', 'pre_beta',
              'size10', 'size12', 'beta12', 'bm12', 'ep13', 'beta10', 'bm10', 'size_beta']
    columns = [column for column in needed if column in formation]
    panel = returns.merge(formation[columns], on=['permno', 'formation_year'], validate='m:1')
    # Surviving observed returns: neither fill missing returns nor impose a
    # future-history/survival requirement at formation.
    return panel.loc[panel.ret.notna()].copy()


def post_ranking_betas(panel, market, group='size_beta'):
    portfolios = panel.groupby(['month', group], observed=True).ret.mean().unstack(group)
    estimates = []
    for portfolio in portfolios:
        beta, se, count = estimate_beta(portfolios[portfolio], market)
        estimates.append({group: portfolio, 'beta': beta, 'beta_se': se, 'months': count})
    result = pd.DataFrame(estimates)
    if result.beta.isna().any():
        raise ValueError('Unidentified post-ranking portfolio beta.')
    return result, portfolios


def fama_macbeth(panel, models=MODELS, winsorize=True, residual_groups=()):
    """Monthly OLS with intercept; conventional FM standard error (no HAC).

    Clip ratios at monthly cross-sectional .5/99.5 percentiles BEFORE logs.
    Portfolio sorts and descriptive statistics retain the original values.
    """
    records, residual_records = [], []
    for month, sample in panel.groupby('month', sort=True):
        sample = sample.copy()
        if winsorize:
            for ratio in ['bm', 'am', 'ab', 'ep_positive']:
                lo, hi = sample[ratio].quantile([.005, .995])
                sample[ratio] = sample[ratio].clip(lo, hi)
                if ratio != 'ep_positive':
                    sample['ln_' + ratio] = np.log(sample[ratio].astype(float))
        for model, variables in models.items():
            valid = sample.dropna(subset=['ret', *variables])
            x = np.column_stack([np.ones(len(valid)), valid[variables].to_numpy(dtype=float)])
            y = valid.ret.to_numpy(dtype=float) * 100
            if len(y) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
                raise ValueError(f'Insufficient cross-section for {model}, {month}')
            coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
            residual = y - x @ coefficients
            for variable, coefficient in zip(['intercept', *variables], coefficients):
                records.append({'month': month, 'model': model, 'variable': variable,
                                'coefficient': coefficient, 'n_observations': len(valid)})
            if residual_groups:
                residual_frame = valid[list(residual_groups)].copy()
                residual_frame['residual'] = residual
                for grouping in residual_groups:
                    for portfolio, average in residual_frame.groupby(grouping).residual.mean().items():
                        residual_records.append({'month': month, 'model': model, 'grouping': grouping,
                                                 'portfolio': portfolio, 'residual': average})
    return pd.DataFrame(records), pd.DataFrame(residual_records)


def summarize_slopes(slopes):
    summary = slopes.groupby(['model', 'variable']).coefficient.agg(['mean', 'std', 'count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['t'] = summary['mean'] / summary['se']
    return summary


def portfolio_summary(panel, group):
    variables = [x for x in ['beta', *CHARACTERISTICS] if x in panel]
    monthly = panel.groupby(['month', group], observed=True)[['ret', *variables]].mean()
    monthly['firms'] = panel.groupby(['month', group], observed=True).size()
    result = monthly.groupby(group).mean()
    result['ret'] *= 100
    return result


def portfolio_matrix(panel, second, variable='ret', market=None):
    """Include stock-weighted All margins, never averages of portfolio averages."""
    def statistic(groups):
        returns = panel.groupby(['month', *groups], observed=True)[variable].mean()
        if market is None:
            result = returns.groupby(groups).mean()
            return result * 100 if variable == 'ret' else result
        wide = returns.unstack(groups)
        return pd.Series({key: estimate_beta(wide[key], market)[0] for key in wide})
    cells = statistic(['size10', second])
    cells.index = pd.MultiIndex.from_tuples(list(cells.index))
    result = cells.unstack().reindex(index=range(1, 11), columns=range(1, 11))
    result.insert(0, 'All', statistic(['size10']).reindex(range(1, 11)))
    all_row = statistic([second]).reindex(range(1, 11))
    all_row.index = list(range(1, 11))
    result.loc['All', range(1, 11)] = all_row
    result.loc['All', 'All'] = (panel.groupby('month')[variable].mean().mean()
                                * (100 if variable == 'ret' else 1)) if market is None else np.nan
    return result.reindex(index=['All', *range(1, 11)])


def form_appendix_sample(crsp, pre_betas):
    december = crsp.loc[(crsp.month.dt.month == 12) & crsp.month.dt.year.between(1940, 1989)
                        & (crsp.exchcd == 1) & crsp.shrcd.isin([10, 11]) & (crsp.me > 0)].copy()
    december['formation_year'] = december.month.dt.year
    december = december.merge(pre_betas, on=['permno', 'formation_year'], validate='1:1')
    december['ln_me'] = np.log(december.me.astype(float))
    output = []
    for _, sample in december.groupby('formation_year'):
        sample = sample.copy()
        sample['size10'] = assign_groups(sample.me, sample.me)
        sample['size12'] = assign_groups(sample.me, sample.me, SPLIT_DECILES)
        sample['beta12'] = assign_groups(sample.pre_beta, sample.pre_beta, SPLIT_DECILES)
        for _, size_group in sample.groupby('size10'):
            sample.loc[size_group.index, 'beta10'] = assign_groups(size_group.pre_beta, size_group.pre_beta)
        sample['size_beta'] = (sample.size10 - 1) * 10 + sample.beta10.astype(int)
        output.append(sample)
    return pd.concat(output, ignore_index=True)

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from core.analysis import (assign_groups, estimate_beta, fama_macbeth, holding_panel,
                           pre_ranking_betas, prepare_accounts, summarize_slopes,
                           portfolio_matrix, form_main_sample, prepare_crsp)


def test_dimson_beta_uses_calendar_lag_and_raw_returns():
    rng = np.random.default_rng(22)
    months = pd.date_range('1960-01-31', periods=80, freq='ME')
    market = pd.Series(rng.normal(.01, .05, 80), index=months)
    returns = .02 + 1.3*market - .2*market.shift(1)
    returns = returns.drop(months[[4, 11, 31]])
    beta, se, n = estimate_beta(returns, market)
    assert_allclose(beta, 1.1, atol=1e-12)
    assert se < 1e-12
    assert n == 76


def test_pre_ranking_window_excludes_future_and_requires_24_pairs():
    rng = np.random.default_rng(21)
    months = pd.date_range('1958-06-30', '1964-06-30', freq='ME')
    market = pd.Series(rng.normal(0, .05, len(months)), index=months)
    returns = .01 + 1.5*market + .4*market.shift(1)
    frame = pd.DataFrame({'month': months, 'permno': 1, 'ret': returns.to_numpy()})
    other = frame.copy()
    other['permno'] = 2
    other.loc[~other.month.isin(pd.date_range('1961-08-31', '1963-06-30', freq='ME')), 'ret'] = np.nan
    frame.loc[frame.month > '1963-06-30', 'ret'] = 1000
    result = pre_ranking_betas(pd.concat([frame, other]), market, 6, 1963, 1963)
    assert result.permno.tolist() == [1]
    assert result.beta_months.tolist() == [60]
    assert_allclose(result.pre_beta, [1.9], atol=1e-11)


def test_quantiles_keep_ties_together_and_use_reference_distribution():
    values = pd.Series([0, 1, 5, 10, 100, np.nan])
    reference = pd.Series(range(1, 11))
    groups = assign_groups(values, reference, [.5])
    assert groups.iloc[:5].tolist() == [1, 1, 1, 2, 2]
    assert pd.isna(groups.iloc[5])


def test_accounts_use_latest_fiscal_end_and_preserve_missing_base_values():
    raw = pd.DataFrame({
        'gvkey': ['a', 'a', 'b', 'c'], 'datadate': pd.to_datetime(['1962-03-31','1962-12-31','1962-12-31','1962-12-31']),
        'popsrc': ['D']*4, 'curcd': ['USD']*4, 'ceq': [10,20,np.nan,15],
        'txdb': [1,2,3,np.nan], 'ib': [2,4,5,np.nan], 'txdi': [1,np.nan,1,1], 'dvp': [0,1,0,0],
    })
    result = prepare_accounts(raw).set_index('gvkey')
    assert result.loc['a','be'] == 22
    assert result.loc['a','earnings'] == 3
    assert result.loc['a','formation_year'] == 1963
    assert pd.isna(result.loc['b','be'])
    assert result.loc['c','be'] == 15
    assert pd.isna(result.loc['c','earnings'])


def test_july_to_june_holding_match_and_missing_returns():
    months = pd.to_datetime(['1963-06-30','1963-07-31','1964-06-30','1964-07-31','1964-08-31'])
    crsp = pd.DataFrame({'permno':1, 'month':months, 'ret':[.1,.2,.3,.4,np.nan]})
    formation = pd.DataFrame({'permno':[1,1], 'formation_year':[1963,1964], 'ln_me':[3,4]})
    panel = holding_panel(crsp, formation)
    assert panel.ln_me.tolist() == [3,3,4]
    assert panel.ret.tolist() == [.2,.3,.4]


def test_monthly_fm_and_conventional_standard_error():
    frames = []
    for i, month in enumerate(pd.date_range('1963-07-31', periods=4, freq='ME')):
        x = np.arange(10, dtype=float)
        frames.append(pd.DataFrame({'month':month, 'ln_me':x, 'ret':(.5+(i+1)*x)/100}))
    slopes, _ = fama_macbeth(pd.concat(frames), models={'size':['ln_me']}, winsorize=False)
    summary = summarize_slopes(slopes).set_index('variable')
    assert_allclose(summary.loc['ln_me','mean'], 2.5)
    assert_allclose(summary.loc['ln_me','se'], np.std([1,2,3,4], ddof=1)/2)
    assert_allclose(summary.loc['intercept','mean'], .5)


def test_all_margin_uses_stocks_not_portfolio_average():
    # Portfolio 1 has one stock, portfolio 2 has nine. The All return is 9%,
    # whereas averaging the two portfolio returns would incorrectly give 5%.
    frame = pd.DataFrame({'month':pd.Timestamp('1963-07-31'), 'size10':1,
                          'beta10':[1]+[2]*9, 'ret':[0]+[.1]*9})
    matrix = portfolio_matrix(frame, 'beta10')
    assert_allclose(matrix.loc[1,'All'], 9)
    assert_allclose(matrix.loc['All','All'], 9)


def test_link_priority_precedes_book_equity_sample_selection():
    n=200
    crsp=pd.DataFrame({'permno':np.tile(np.arange(n),2),
        'date':np.repeat(pd.to_datetime(['1962-12-31','1963-06-30']),n),
        'prc':np.tile(np.arange(n)+10.,2),'shrout':1000.,'ret':.01,
        'exchcd':1,'shrcd':10,'siccd':2000})
    crsp=prepare_crsp(crsp)
    pre=pd.DataFrame({'permno':np.arange(n),'formation_year':1963,
                      'pre_beta':np.linspace(.5,2,n),'beta_months':60})
    accounts=pd.DataFrame({'gvkey':[str(i) for i in range(n)]+['secondary'],
        'datadate':pd.Timestamp('1962-12-31'),'popsrc':'D','curcd':'USD',
        'ceq':[-1]+list(np.arange(n-1)+10.)+[50.], 'txdb':0.,'txditc':0.,
        'ib':1.,'txdi':0.,'dvp':0.,'at':100.,'sich':2000})
    links=pd.DataFrame({'gvkey':[str(i) for i in range(n)]+['secondary'],
        'permno':list(range(n))+[0],'linkdt':pd.Timestamp('1960-01-01'),
        'linkenddt':pd.NaT,'linktype':'LC','linkprim':['P']*n+['C']})
    formation,audit=form_main_sample(crsp,prepare_accounts(accounts),links,pre)
    assert 0 not in formation.permno.to_list()
    assert audit['overlapping_link_rows']==1


def test_market_summary_convention_does_not_transform_fm_slopes():
    from core.pipeline import subperiod_summary
    months=pd.date_range('1963-07-31',periods=3,freq='ME')
    slopes=pd.DataFrame({'month':months,'model':'size','variable':'ln_me','coefficient':[-1.,-2.,-3.]})
    market=pd.DataFrame({'vwretd':[.1,-.1,.05],'ewretd':[.1,-.1,.05]},index=months)
    periods=[('sample','1963-07-31','1963-09-30')]
    log=subperiod_summary(slopes,market,periods)
    arithmetic=subperiod_summary(slopes,market,periods,False)
    assert_allclose(log.loc[log.variable=='vwretd','mean'],np.log1p([.1,-.1,.05]).mean()*100)
    assert_allclose(arithmetic.loc[arithmetic.variable=='vwretd','mean'],np.mean([.1,-.1,.05])*100)
    assert_allclose(log.loc[log.variable=='ln_me','mean'],[-2.])

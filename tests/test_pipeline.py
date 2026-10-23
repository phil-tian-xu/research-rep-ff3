"""Small synthetic end-to-end check; these inputs never enter real outputs."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import core.pipeline as pipeline
import core.report as report


def test_complete_pipeline_and_report(tmp_path, monkeypatch):
    rng = np.random.default_rng(1992)
    raw = tmp_path/'data/raw'
    raw.mkdir(parents=True)
    (tmp_path/'config').mkdir()
    shutil.copy(Path(__file__).resolve().parents[1]/'config/paper_benchmarks.json',tmp_path/'config/paper_benchmarks.json')
    months = pd.date_range('1935-12-31','1990-12-31',freq='ME')
    market = rng.normal(.008,.04,len(months))
    n = 200
    permnos = np.arange(n)+10000
    crsp = pd.DataFrame({'permno':np.repeat(permnos,len(months)),
        'date':np.tile(months,n),'prc':np.repeat(np.exp(np.linspace(1,8,n)),len(months)),
        'shrout':1000.,'exchcd':1,'shrcd':10,'siccd':2000,
        'ret':np.tile(market,n)*np.repeat(rng.uniform(.5,1.8,n),len(months))+rng.normal(0,.03,n*len(months))})
    crsp.to_parquet(raw/'crsp_monthly.parquet',index=False)
    records=[]
    for i in range(n):
        for year in range(1962,1990):
            be=np.exp(np.linspace(1,8,n))[i]*rng.uniform(.2,2)
            records.append({'gvkey':str(i),'datadate':pd.Timestamp(year,12,31),
                'popsrc':'D','curcd':'USD','sich':2000,'ceq':be,'txdb':0.,'txditc':0.,
                'at':be*rng.uniform(1.2,2),'ib':be*rng.uniform(-.1,.3),'txdi':0.,'dvp':0.})
    pd.DataFrame(records).to_parquet(raw/'compustat_annual.parquet',index=False)
    pd.DataFrame({'gvkey':[str(i) for i in range(n)],'permno':permnos,
        'linkdt':pd.Timestamp('1935-01-01'),'linkenddt':pd.NaT,
        'linkprim':'P','linktype':'LC'}).to_parquet(raw/'ccm_links.parquet',index=False)
    for name in ['market_returns','nyse_market']:
        pd.DataFrame({'date':months,'vwretd':market,'ewretd':market+.001}).to_parquet(raw/f'{name}.parquet',index=False)
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'RAW',raw)
    monkeypatch.setattr(report,'ROOT',tmp_path)
    monkeypatch.setattr(report,'RAW',raw)
    results=pipeline.run_replication(tmp_path/'results')
    report.write_report(tmp_path/'results',results)
    assert results['audit']['months']==330
    assert results['audit']['appendix_months']==600
    assert len(results['regression_comparison'])==26
    assert (tmp_path/'results/report.html').is_file()
    assert (tmp_path/'results/appendix/table_aiv_subperiods.csv').is_file()
    assert results['matrices']['table_v_returns'].shape==(11,11)

from zipfile import ZipFile

import pandas as pd
import pytest

from core.report import read_french_equal_weighted


@pytest.mark.parametrize('heading', ['Average Equal Weighted Returns -- Monthly',
                                    'Equal Weight Returns -- Monthly',
                                    'Average Equal Weight Returns -- Monthly'])
def test_monthly_french_block_preserves_missing_and_excludes_annual(tmp_path,heading):
    rows=[heading,',small,big']
    for i,month in enumerate(pd.date_range('1963-07-31','1990-12-31',freq='ME')):
        rows.append(f'{month:%Y%m},{-99.99 if i==0 else 1.25},2.50')
    rows.extend(['','Equal Weight Returns -- Annual','1964,99,99'])
    path=tmp_path/'example.zip'
    with ZipFile(path,'w') as archive:
        archive.writestr('example.csv','\n'.join(rows))
    frame=read_french_equal_weighted(path)
    assert frame.shape==(330,2)
    assert pd.isna(frame.iloc[0,0])
    assert frame.iloc[-1,0]==1.25  # Keep source percent units.
    assert frame.big.mean()==2.50

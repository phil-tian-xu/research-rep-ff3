"""Immutable, dated WRDS extracts and optional Kenneth French benchmarks."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text, URL


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
QUERIES = {
    'crsp_monthly': """
select a.permno, a.permco, a.date, a.ret, a.prc, a.shrout,
       b.exchcd, b.shrcd, b.siccd
from crsp.msf a
left join crsp.msenames b on a.permno=b.permno
    and b.namedt <= a.date and a.date <= coalesce(b.nameendt, date '9999-12-31')
where a.date between date '1935-12-01' and date '1990-12-31'
order by a.permno, a.date
""",
    'compustat_annual': """
select gvkey, datadate, fyear, fyr, popsrc, curcd, sich,
       at, ceq, txdb, txditc, ib, txdi, dvp
from comp.funda
where indfmt='INDL' and datafmt='STD' and consol='C'
  and datadate between date '1962-01-01' and date '1989-12-31'
order by gvkey, datadate, popsrc
""",
    'ccm_links': """
select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim, liid
from crsp.ccmxpf_linktable
where lpermno is not null and linktype in ('LC','LU','LS')
order by gvkey, lpermno, linkdt nulls first, linkenddt nulls first
""",
    'market_returns': """
select date, vwretd, ewretd
from crsp.msi
where date between date '1935-12-01' and date '1990-12-31'
order by date
""",
    'nyse_market': """
select caldt as date, vwretd, ewretd
from crsp_a_indexes.msia
where caldt between date '1935-12-01' and date '1990-12-31'
order by caldt
""",
}


def download_wrds(refresh=False, datasets=None):
    """Use libpq's saved password; never prompt for, print, or save credentials."""
    RAW.mkdir(parents=True, exist_ok=True)
    pending = []
    for name, sql in QUERIES.items():
        if datasets is not None and name not in datasets:
            continue
        path = RAW / f'{name}.parquet'
        metadata_path = path.with_suffix('.json')
        if path.exists() and metadata_path.exists() and not refresh:
            info = json.loads(metadata_path.read_text())
            if info['specification_sha256'] != sha256(sql.encode()).hexdigest():
                raise ValueError(f'{name}: query changed; use --refresh-data explicitly.')
            if pq.read_metadata(path).num_rows != info['rows']:
                raise ValueError(f'{name}: cache row count does not match provenance.')
            print(f'{name}: cached {info["rows"]:,} rows', flush=True)
        else:
            pending.append((name, sql))
    if not pending:
        return
    username = os.environ.get('WRDS_USERNAME')
    if not username:
        raise RuntimeError('Set WRDS_USERNAME and configure the PostgreSQL password file.')
    engine = create_engine(URL.create(
        'postgresql+psycopg2', username=username, host='wrds-pgdata.wharton.upenn.edu',
        port=9737, database='wrds'), connect_args={'sslmode': 'require', 'connect_timeout': 30})
    try:
        with engine.connect() as connection:
            for name, sql in pending:
                print(f'{name}: querying WRDS...', flush=True)
                path = RAW / f'{name}.parquet'
                temporary = path.with_suffix('.partial.parquet')
                count = 0
                writer = None
                try:
                    stream = connection.execution_options(stream_results=True)
                    for frame in pd.read_sql_query(text(sql), stream, chunksize=100_000):
                        if frame.empty:
                            continue
                        for column in ['date', 'datadate', 'linkdt', 'linkenddt']:
                            if column in frame:
                                frame[column] = pd.to_datetime(frame[column])
                        for column in ['permno', 'permco', 'exchcd', 'shrcd', 'siccd', 'sich', 'fyear', 'fyr']:
                            if column in frame:
                                frame[column] = frame[column].astype('Int64')
                        table = pa.Table.from_pandas(frame, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                        writer.write_table(table)
                        count += len(frame)
                        print(f'{name}: {count:,} rows', flush=True)
                finally:
                    if writer is not None:
                        writer.close()
                if not count:
                    raise ValueError(f'{name}: empty WRDS extract')
                temporary.replace(path)
                info = {
                    'sql': sql, 'specification_sha256': sha256(sql.encode()).hexdigest(),
                    'file_sha256': sha256(path.read_bytes()).hexdigest(), 'rows': count,
                    'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                    'source': 'WRDS, CRSP legacy FIZ / Compustat North America / CCM',
                    'return_units': 'decimal', 'market_equity_units': 'USD millions = abs(prc)*shrout/1000',
                    'delisting_adjustment': 'none; original CRSP RET, no additional DLRET merge',
                }
                path.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n')
    finally:
        engine.dispose()


def prepare_with_reference(reference_project):
    """Read existing extracts without importing or changing the other project.

    Only early CRSP history, dated names, and paper-specific accounting fields
    are requested from WRDS. The source cache's SQL and checksum are retained.
    """
    reference = Path(reference_project)/'data/raw/lsv1994/nyse_amex_nasdaq_19570101_20241231_comp19570101_20241231'
    if not (reference/'crsp_monthly.parquet').is_file():
        raise FileNotFoundError(f'Expected reference extract: {reference}')
    RAW.mkdir(parents=True, exist_ok=True)
    username = os.environ.get('WRDS_USERNAME')
    if not username:
        raise RuntimeError('Set WRDS_USERNAME and configure a PostgreSQL password file.')
    engine = create_engine(URL.create('postgresql+psycopg2', username=username,
        host='wrds-pgdata.wharton.upenn.edu', port=9737, database='wrds'),
        connect_args={'sslmode':'require','connect_timeout':30})
    extra = {
        'early_crsp': "select permno, permco, date, ret, prc, shrout from crsp.msf where date between date '1935-12-01' and date '1956-12-31'",
        'names': "select permno, permco, namedt, nameendt, exchcd, shrcd, siccd from crsp.msenames where namedt <= date '1990-12-31' and coalesce(nameendt,date '9999-12-31') >= date '1935-12-01'",
        'compustat_annual':QUERIES['compustat_annual'], 'market_returns':QUERIES['market_returns'],
        'nyse_market':QUERIES['nyse_market'],
    }
    extracted = {}
    try:
        with engine.connect() as connection:
            for name, sql in extra.items():
                path = RAW/f'{name}.parquet'
                if path.exists() and path.with_suffix('.json').exists():
                    metadata = json.loads(path.with_suffix('.json').read_text())
                    if metadata.get('sql') == sql:
                        print(f'{name}: cached', flush=True)
                        extracted[name] = path
                        continue
                    raise ValueError(f'{name} source query changed; preserve or remove its old cache explicitly.')
                print(f'{name}: downloading missing fields/history...', flush=True)
                frames = []
                for chunk in pd.read_sql_query(text(sql), connection.execution_options(stream_results=True), chunksize=10_000):
                    frames.append(chunk)
                    print(f'{name}: batch {len(frames)}, {sum(len(x) for x in frames):,} rows', flush=True)
                frame = pd.concat(frames, ignore_index=True)
                for column in ['date','namedt','nameendt','datadate']:
                    if column in frame:
                        frame[column] = pd.to_datetime(frame[column])
                frame.to_parquet(path,index=False,compression='zstd')
                path.with_suffix('.json').write_text(json.dumps({
                    'sql':sql, 'specification_sha256':sha256(QUERIES.get(name,sql).encode()).hexdigest(),
                    'file_sha256':sha256(path.read_bytes()).hexdigest(), 'rows':len(frame),
                    'retrieved_at_utc':datetime.now(timezone.utc).isoformat(), 'source':'WRDS legacy CRSP / Compustat',
                },indent=2)+'\n')
                extracted[name] = path
    finally:
        engine.dispose()
    assemble_reference_history(reference_project)


def assemble_reference_history(reference_project):
    """Assemble already-extracted histories; never write to the reference path."""
    reference=Path(reference_project)/'data/raw/lsv1994/nyse_amex_nasdaq_19570101_20241231_comp19570101_20241231'
    path = RAW/'crsp_monthly.parquet'
    if not path.exists():
        print('Combining read-only reference history with the early extract...', flush=True)
        source = reference/'crsp_monthly.parquet'
        base = pd.read_parquet(source, filters=[('date','<=',pd.Timestamp('1990-12-31'))],
                               columns=['permno','date','ret','prc','shrout'])
        early = pd.read_parquet(RAW/'early_crsp.parquet').drop(columns='permco')
        base = pd.concat([early,base], ignore_index=True)
        names = pd.read_parquet(RAW/'names.parquet')
        base['permno'] = base.permno.astype('int64')
        names['permno'] = names.permno.astype('int64')
        base = pd.merge_asof(base.sort_values('date'),names.sort_values('namedt'),
                            by='permno',left_on='date',right_on='namedt',direction='backward')
        outside = base.nameendt.notna() & (base.date > base.nameendt)
        base.loc[outside,['exchcd','shrcd','siccd','permco']] = None
        base = base.drop(columns=['namedt','nameendt']).sort_values(['permno','date'])
        if base.duplicated(['permno','date']).any():
            raise ValueError('Overlapping source histories produced duplicate CRSP keys.')
        base.to_parquet(path,index=False,compression='zstd')
        path.with_suffix('.json').write_text(json.dumps({
            'construction':'Reference CRSP 1957-1990 plus WRDS 1935-1956, enriched with dated CRSP names',
            'reference_path':str(source),'reference_sha256':sha256(source.read_bytes()).hexdigest(),
            'reference_metadata':json.loads(source.with_suffix('.json').read_text()),
            'specification_sha256':sha256(QUERIES['crsp_monthly'].encode()).hexdigest(),
            'file_sha256':sha256(path.read_bytes()).hexdigest(),'rows':len(base),
            'return_units':'decimal','delisting_adjustment':'none; legacy CRSP RET',
        },indent=2)+'\n')
    path = RAW/'ccm_links.parquet'
    if not path.exists():
        source = reference/path.name
        path.write_bytes(source.read_bytes())
        path.with_suffix('.json').write_text(json.dumps({
            'construction':'Unmodified read-only reference CCM extract',
            'reference_path':str(source),'reference_sha256':sha256(source.read_bytes()).hexdigest(),
            'reference_metadata':json.loads(source.with_suffix('.json').read_text()),
            'specification_sha256':sha256(QUERIES['ccm_links'].encode()).hexdigest(),
            'file_sha256':sha256(path.read_bytes()).hexdigest(),'rows':pq.read_metadata(path).num_rows,
        },indent=2)+'\n')


def download_french(refresh=False):
    """Save source ZIPs as supplementary, differently constructed benchmarks."""
    folder = RAW / 'french'
    folder.mkdir(exist_ok=True)
    for filename in ['Portfolios_Formed_on_ME_CSV.zip', 'Portfolios_Formed_on_BE-ME_CSV.zip',
                     '100_Portfolios_10x10_CSV.zip', 'F-F_Research_Data_Factors_CSV.zip']:
        path = folder / filename
        if path.exists() and not refresh:
            metadata_path=path.with_suffix('.json')
            if not metadata_path.exists() or json.loads(metadata_path.read_text())['sha256'] != sha256(path.read_bytes()).hexdigest():
                raise ValueError(f'French snapshot checksum/provenance mismatch: {filename}')
            continue
        url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/' + filename
        print(f'French benchmark: {filename}', flush=True)
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={'User-Agent':'Mozilla/5.0',
                            'Accept-Encoding':'identity','Connection':'close'}), timeout=60) as response:
                    chunks=[]
                    while chunk := response.read(65536):
                        chunks.append(chunk)
                    content=b''.join(chunks)
                with ZipFile(BytesIO(content)) as archive:
                    if archive.testzip() is not None:
                        raise ValueError(f'Invalid ZIP checksum for {url}')
                break
            except (OSError, ValueError, BadZipFile) as error:
                if attempt==2:
                    raise
                print(f'French transfer retry {attempt+1}: {type(error).__name__}',flush=True)
                sleep(2)
        path.write_bytes(content)
        path.with_suffix('.json').write_text(json.dumps({
            'source_url': url, 'sha256': sha256(content).hexdigest(),
            'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'role': 'supplementary benchmark; not the original 1992 portfolio definitions',
        }, indent=2) + '\n')


def validate_raw_inputs():
    """Fail before estimation if a required raw snapshot/provenance is damaged."""
    for name in QUERIES:
        path = RAW/f'{name}.parquet'
        metadata_path = path.with_suffix('.json')
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f'Missing {name} extract or provenance. Run --prepare-only first.')
        metadata = json.loads(metadata_path.read_text())
        if pq.read_metadata(path).num_rows != metadata['rows']:
            raise ValueError(f'{name}: raw row count differs from provenance.')
        if sha256(path.read_bytes()).hexdigest() != metadata['file_sha256']:
            raise ValueError(f'{name}: raw checksum differs from provenance.')



"""Run the FF1992 replication. See README for methodology and source choices."""

import argparse
from pathlib import Path

from core.data import ROOT, download_french, download_wrds, prepare_with_reference, validate_raw_inputs
from core.pipeline import run_replication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true', help='Download inputs without running analysis.')
    parser.add_argument('--skip-download', action='store_true', help='Run entirely from existing raw extracts.')
    parser.add_argument('--refresh-data', action='store_true', help='Explicitly replace cached raw snapshots.')
    parser.add_argument('--skip-appendix', action='store_true', help='Run main Tables I-VI only.')
    parser.add_argument('--tax-variable', choices=['txdb','txditc'], default='txdb',
                        help='Book equity deferred-tax field; txdb matches the paper definition.')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--reference-project', type=Path, help='Read cached stock history from this project without modifying it.')
    args = parser.parse_args()
    if not args.skip_download:
        if args.reference_project:
            if args.refresh_data:
                parser.error('--refresh-data cannot be combined with --reference-project; use a fresh raw folder.')
            prepare_with_reference(args.reference_project)
        else:
            download_wrds(args.refresh_data)
        download_french(args.refresh_data)
    if args.prepare_only:
        return
    validate_raw_inputs()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    results = run_replication(output, not args.skip_appendix, args.tax_variable)
    from core.report import write_report
    write_report(output, results)
    print(f'Finished. Report: {output / "report.html"}', flush=True)
    print(results['regression_comparison'].loc[
        results['regression_comparison'].model.isin(['beta','size','size_bm']),
        ['model','variable','paper_mean','mean','paper_t','t']].to_string(index=False))


if __name__ == '__main__':
    main()

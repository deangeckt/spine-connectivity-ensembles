"""
Populate `data/` from the Zenodo snapshot — the fast route, minutes not days.

This is route 1 of the two ways to get the structural data behind figures 1-5 (and the
structural half of 6, 7, S1-S15). `download_data_cave.py` is route 2: it rebuilds the
same tables from the MICrONS source with CAVEclient, takes about a day, and needs a CAVE
account. Both are pinned to materialization v1718 and produce the same files, so pick
whichever you prefer and run only that one.

What it downloads

    https://zenodo.org/records/21104426  —  data.zip, 1.4 GB compressed, 4.4 GB unpacked
    DOI 10.5281/zenodo.21104425 (all versions) / 10.5281/zenodo.21104426 (this one)

and unpacks it into `data/` next to this script:

    data/raw_tables/               two of the seven CAVE tables — the column's manual
                                   cell types and the proofreading status, which are the
                                   two the figures read
    data/micro_column_network/     the 1,351-cell column: neurons, skeletons, meshes,
                                   synapse and spine tables, connectivity matrix,
                                   subnetworks
    data/all_axon_pr_network/      the proofread-axon network — figure S3

The archive also carries the pre-rendered dendrite images and EM cut-outs behind figures
1, 2 and S1, under `data/figures/spine_project/`. Nothing reads them any more: neither
download route can rebuild those images, so they ship in the repository itself, in
`images/`, and that is where the notebooks look. The extracted copy is redundant and can
be deleted.

What it does NOT download

    data/activity/                 the MICrONS calcium recordings (~19 GB) and the
                                   stimulus tables, behind figures 6, 7, S14 and S15.
                                   Too large to redistribute, so neither route ships
                                   them — build them with
                                   scripts/extract_calcium_data_via_docker.ipynb, then
                                   run scripts/ensemble_run.py. See the README.

Two things live only on the CAVE route. The first is the other five raw tables — among
them `coregistration_manual_v4.csv`, which scripts/extract_calcium_data_via_docker.ipynb
needs; `python download_data_cave.py --steps raw` fetches all seven in a couple of
minutes and needs a CAVE account. The second: the archive carries six of the ten cell
meshes. The four it lacks are four of figure 7's six cells, and that notebook ships set
to draw its cells as skeletons, so nothing breaks without them.

Usage

    python download_data_zenodo.py                # download, extract, keep the archive
    python download_data_zenodo.py --delete-zip   # ...then delete data.zip
    python download_data_zenodo.py --force        # re-download over an existing data/

Safe to re-run: it stops if `data/` already looks populated (use --force to override),
and reuses a complete `data.zip` that is already on disk rather than fetching it twice.
"""

import argparse
import os
import sys
import urllib.request
import zipfile

# The progress bar below draws a block character, and the messages use arrows and dashes;
# a Windows console on a non-UTF-8 code page raises UnicodeEncodeError on them mid-run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

# The data deposit, not the code archive: 10.5281/zenodo.21104860 in the README badge is
# the citable snapshot of *this repository*, a different record.
ZENODO_URL = 'https://zenodo.org/records/21104426/files/data.zip?download=1'

# Everything resolves against the script's own folder, so the script works from any cwd.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(REPO_ROOT, 'data.zip')
DATA_DIR = os.path.join(REPO_ROOT, 'data')

# Present in the archive; used to tell a populated data/ from an empty or half-made one.
EXPECTED_DIRS = ['raw_tables', 'micro_column_network', 'all_axon_pr_network']


def show_progress(block_num, block_size, total_size):
    """urlretrieve reporthook — a one-line progress bar that overwrites itself."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)

        bar_length = 50
        filled_length = int(bar_length * downloaded // total_size)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)

        # \r forces the cursor to the start of the line to overwrite it
        sys.stdout.write(f'\rDownloading: [{bar}] {percent:.1f}% '
                         f'({downloaded_mb:.1f} MB / {total_mb:.1f} MB)')
        sys.stdout.flush()


def remote_size():
    """Content-Length of the Zenodo file, or None if the server will not say."""
    try:
        with urllib.request.urlopen(
                urllib.request.Request(ZENODO_URL, method='HEAD'), timeout=60) as response:
            length = response.headers.get('Content-Length')
            return int(length) if length else None
    except Exception as e:
        print(f'Could not check the remote file size ({type(e).__name__}: {e}) — '
              f'continuing anyway.')
        return None


def already_populated():
    return all(os.path.isdir(os.path.join(DATA_DIR, d)) for d in EXPECTED_DIRS)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='download and extract even if data/ is already populated')
    parser.add_argument('--delete-zip', action='store_true',
                        help='delete data.zip once it has been extracted')
    args = parser.parse_args()

    if already_populated() and not args.force:
        print(f'data/ already holds {", ".join(EXPECTED_DIRS)} — nothing to do.')
        print('Re-run with --force to download and overwrite it.')
        return

    expected = remote_size()

    # A complete archive from an earlier run is reused; a truncated one (an interrupted
    # download) is thrown away and fetched again.
    if os.path.exists(ZIP_PATH) and expected and os.path.getsize(ZIP_PATH) == expected:
        print(f'Reusing the data.zip already on disk ({expected / 1e9:.2f} GB).')
    else:
        if os.path.exists(ZIP_PATH):
            print('The data.zip on disk is incomplete — downloading it again.')
        print('Initializing download from Zenodo...')
        urllib.request.urlretrieve(ZENODO_URL, ZIP_PATH, reporthook=show_progress)
        print('\nDownload complete.')

    print('Extracting files...')
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(REPO_ROOT)

    if args.delete_zip:
        os.remove(ZIP_PATH)
        print('Deleted data.zip.')
    else:
        print(f'Extraction complete. You can delete {ZIP_PATH} to save '
              f'{os.path.getsize(ZIP_PATH) / 1e9:.1f} GB.')

    missing = [d for d in EXPECTED_DIRS if not os.path.isdir(os.path.join(DATA_DIR, d))]
    if missing:
        print(f'!! Warning: the archive did not contain: {", ".join(missing)}')
        return

    print(f"Success! All data files are now available in '{DATA_DIR}'.")
    print('Figures 1-5 and S1-S13 will run as they are. Figures 6, 7, S14 and S15 also '
          'need data/activity/ — see the README.')


if __name__ == '__main__':
    main()

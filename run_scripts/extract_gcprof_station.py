
#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Extract tracer profiles and compute total columns
above a given station from GEOS-Chem outputs 

This script can run on a IPython parallel cluster, if
available.

"""

import os
import string
import argparse
import getpass
import logging
import sys
import time
import datetime
import subprocess
import inspect
from glob import iglob


# ---------------------------------------
# Find and execute the 'process' script.
# ---------------------------------------

this_script = inspect.getframeinfo(inspect.currentframe()).filename
this_script_dir = os.path.dirname(os.path.abspath(this_script))
process_script = os.path.join(this_script_dir,
                              'extract_gcprof_station_process.py')

execfile(process_script)


# ---------------------------------------
# Functions related to logging
# ---------------------------------------

def set_logger(logfile, loglevel):
    """Set the logger."""
    
    logger = logging.getLogger('ProfStationGC')
    
    if not isinstance(loglevel, int):
        loglevel = getattr(logging, loglevel.upper(), None)
    logger.setLevel(loglevel)
    
    log_formatter = logging.Formatter(
        '%(levelname)s %(asctime)s [%(name)s] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )
    
    if logfile is None:
        log_handler = logging.StreamHandler(stream=sys.stdout)
    else:
        log_handler = logging.FileHandler(logfile, mode='w')
    log_handler.setFormatter(log_formatter)
    
    logger.addHandler(log_handler)
    logger.propagate = False
    
    return logger

    
def wait_log_return(ar, logger, dt=1, dtlog=30, truncate=100000):
    """
    Wait for IPython parallel AsyncResult `ar` while
    logging progress. Then return the results.
    """
    def log_ar_progress():
        tdelta = datetime.timedelta(seconds=ar.elapsed)
        logger.info("{} files processed - time elapsed {}"
                    .format(ar.progress, tdelta))
        
    tlog = time.time()
    while not ar.ready():
        t = time.time()
        if t - tlog >= dtlog:
            tlog = t
            log_ar_progress()
            # TODO: every x times, get current results,
            # merge, and save in temp file(s)
        time.sleep(dt)
    log_ar_progress()
    
    for metadata, stdout in zip(ar.metadata, ar.stdout):
        if stdout:
            logger.debug("Engine #{} output:\n{}".
                         format(metadata['engine_id'], stdout[-truncate:]))
    
    return ar.result()


# ---------------------------------------
# Functions for start/stop IPython parallel cluster
# ---------------------------------------

def start_ipcluster(ipcluster_exe, nengines, profile,
                        max_retries=50):
    """
    Start a new IPython parallel cluster (daemon)
    with a number of `nengines` and using `profile`.
    """
    from ipyparallel import Client

    ipcluster = None
    rc = None
    dview = None
    lview = None

    ipcluster = os.system(
        '{} start -n={} --profile={} --daemon'
        .format(ipcluster_exe, nengines, profile)
    )

    # retry until ipcluster is ready
    time.sleep(3)
    rc = Client(profile=profile)
    retries = 0
    while True:
        if retries > max_retries:
            stop_ipcluster(ipcluster_exe, profile)
            raise Exception("impossible to access to (all) engines "
                            "of the IPython parallel cluster")
        if len(rc.ids) < nengines:
            retries += 1
            time.sleep(1)
            continue
        else:
            break

    dview = rc[:]
    lview = rc.load_balanced_view()

    return ipcluster, rc, dview, lview


def stop_ipcluster(ipcluster_exe, profile):
    """
    Stop the IPython parallel cluster.
    """
    os.system('{} stop --profile={}'
              .format(ipcluster_exe, profile))


# ---------------------------------------
# Main functions
# ---------------------------------------

def main(args, dview=None, lview=None, logger=None):
    """
    Main function.
    
    Include parsing inputs/params,
    calling to processing functions, launch serial or
    parallel tasks, logging...
    """
    
    if dview is not None and lview is not None:
        parallel = True
    else:
        parallel = False
    
    # IMPORT PARAMS FROM INPUT FILE
    logger.info("Loading the parameters/arguments from file '{}'..."
                .format(os.path.abspath(args.infile)))
    
    try:
        params = load_params(args.infile)
        
        in_files = parse_in_files(params['in_dir'],
                                  params['in_files'])
        
        logger.debug('Input files {}'.format(in_files))
        
        if not len(in_files):
            logger.error('No file to process')
            return
        
    except Exception:
        logger.exception('Error while loading or parsing the file')
    
    # PRE-PROCESSING
    logger.info('Pre-processing...')
    
    if parallel:
        dview.block = True
        # execute process script and preprocessing func in all engines
        dview.execute("execfile('{}')".format(process_script))
        dview.apply_sync(lambda p=params: preprocessing(p))
        # import modules, functions, variables in all engines
        #dview.execute('import warnings')
        #dview.execute('warnings.filterwarnings("ignore")')
    
    else:
        preprocessing(params)
    
    # PROCESSING
    logger.info('Loading data, regridding and calculating columns ({} file(s))...'
                .format(len(in_files)))
    
    if parallel:
        lview.block = False
        ar = lview.map(lambda ifile, p=params:
                           processing(ifile, p),
                       in_files)
        res = wait_log_return(ar, logger, dtlog=30)
        
        logger.info('Merging the results...')
        regridded_profiles, tracer_columns = merge_results(res)
    
    else:
        regridded_profiles, tracer_columns = processing(in_files,
                                                        params)
        
    logger.info('Profiles summary (cubes):\n\n{}\n'
                .format(regridded_profiles))
    logger.info('Columns summary (cubes):\n\n{}\n'
                .format(tracer_columns))
        
    # SAVE OUTPUTS
    logger.info('Writing output files (this may take a while)...')
    
    data_desc, written_files = save_results(regridded_profiles,
                                            tracer_columns,
                                            params)

    logger.info('Profiles summary (tables):\n\n{}\n'
                .format(data_desc[0]))
    logger.info('Columns summary (tables):\n\n{}\n'
                    .format(data_desc[1]))
    for f in written_files:
        logger.info('Written file {}'.format(f))

    logger.info('Finished')    
  

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Extract and regrid profiles at a given station from GEOS-Chem outputs'
    )
    parser.add_argument('infile', nargs='?',
                        help='input arguments or parameters file (yaml format)')
    parser.add_argument('--logfile', nargs='?',
                        default=None,
                        help='path to the log file (default: stdout/stderr)')
    parser.add_argument('--loglevel', nargs='?',
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
                                 50, 40, 30, 20, 10, 0],
                        help='log level')
    parser.add_argument('--nengines', type=int, default=1,
                        help='number of nodes to use (IPython parallel engines)')
    args = parser.parse_args()
    
    logger = set_logger(args.logfile, args.loglevel)
    
    logger.info('Working directory is: ' + os.getcwd())
    
    user = getpass.getuser()
    ipy_profile = 'nb_{}'.format(user)
    ipcluster_exe = os.path.join(
        os.path.dirname(sys.executable),
        'ipcluster'
    )
    
    if args.nengines > 1:
        logger.warn("Starting an IPython parallel cluster "
                    "with {} engines and using profile '{}'..."
                    .format(args.nengines, ipy_profile))
        logger.info("Waiting for the cluster "
                    "(this may take a few seconds)...")
        
        ipcluster_daemon, rc, dview, lview = start_ipcluster(
            ipcluster_exe, args.nengines, ipy_profile
        )
        
        # Uncomment below if using an existing cluster
        #from IPython.parallel import Client
        #rc = Client(profile=ipy_profile)
        #dview = rc[:]
        #lview = rc.load_balanced_view()
    
    else:
        logger.warn("No IPython parallel cluster will be used")
        
        ipcluster_daemon = None
        dview = None
        lview = None
    
    try:
        main(args, dview=dview, lview=lview, logger=logger)
    except Exception:
        logger.critical("Process interrupted!")
        raise
    finally:
        if ipcluster_daemon is not None:
            logger.warning("Shutting down the IPython parallel cluster...")
            stop_ipcluster(ipcluster_exe, ipy_profile)
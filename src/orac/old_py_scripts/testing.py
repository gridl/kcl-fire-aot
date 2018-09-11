#!/usr/bin/env python2.7
# Run ORAC regression tests.
# 29 Feb 2016, AP: Finalised initial version.
# 27 Jun 2016, AP: P2.7 rewrite
# 08 Jul 2016, AP: Debugging against more awkward python environments

from colours import cprint

import argparse
import glob
import local_defaults as defaults
import orac_utils as ou
import os
import subprocess
import tempfile
import warnings

warnings.filterwarnings('always', category=ou.OracWarning)

# First lets define parser that will hold all of the processing variables
# and that can be updated at a later stage.

parser = argparse.ArgumentParser(
    description='Run the full ORAC suite on a given Level 1B file.')
#parser.add_argument('-A', '--all_phases', action='store_true',
#                     help = 'Sets phases to run all possible tests.')

ou.args_common(parser)
ou.args_preproc(parser)
ou.args_main(parser)
ou.args_postproc(parser)
ou.args_cc4cl(parser)
args = parser.parse_args()


target = 'MYD021KM.A2008172.0405.005.2009317014309.hdf'
limit = (800, 900, 1000, 1100)


args.target = target
args.limit = limit

args.in_dir = [ defaults.data_dir + '/testinput' ]
base_out_dir = args.out_dir
base_out_dir = defaults.data_dir +'/testoutput'
args.out_dir = base_out_dir + '/modis_testing'


try:
    # Run ORAC suite
    ou.check_args_cc4cl(args)
    (fileroot, dirs, jid) = ou.cc4cl(args)

except ou.OracError as err:
    cprint('ERROR) ' + err.message, ou.colouring['error'])
except KeyboardInterrupt:
    cprint('Execution halted by user.', ou.colouring['error'])
except subprocess.CalledProcessError as err:
    cprint('{:s} failed with error code {:d}. {:s}'.format(
        ' '.join(err.cmd), err.returncode, err.output))

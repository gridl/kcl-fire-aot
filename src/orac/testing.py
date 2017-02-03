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



target = 'MYD021KM.A2008172.0405.005.2009317014309.hdf'
limit = (0, 0, 0, 0)


# Define parser
parser = argparse.ArgumentParser(description='Run ORAC regression tests.')
reg = parser.add_argument_group('Regression test parameters')
reg.add_argument('-A', '--all_phases', action='store_true',
                 help = 'Sets phases to run all possible tests.')
reg.add_argument('-L', '--long', action='store_true',
                 help = 'Process full orbits rather than short segments.')
reg.add_argument('-t', '--tests', type=str, nargs='+', metavar='TEST',
                 choices = regress.keys(),
                 default = ['DAYMYDS', 'NITMYDS', 'DAYAATSRS', 'NITAATSRS',
                            'DAYAVHRRS', 'NITAVHRRS'],
                 help = 'List of tests to run.')

ou.args_common(parser, regression=True)
ou.args_preproc(parser)
ou.args_main(parser)
ou.args_postproc(parser)
ou.args_cc4cl(parser)
args = parser.parse_args()

# Lets just run over MODIS
args.tests = ['DAYMYD']

if args.all_phases:
    args.phases = ou.settings.keys()

if args.in_dir == None:
    args.in_dir = [ defaults.data_dir + '/testinput' ]
if args.out_dir:
    base_out_dir = args.out_dir
else:
    base_out_dir = defaults.data_dir +'/testoutput'

try:
    for test in args.tests:
        cprint(test, ou.colouring['header'])

        # Set filename to be processed and output folder
        args.target  = regress[test][0]  # for regression test file to be processes is set here
                                         # in normal running it is set in args_common
        args.limit   = regress[test][1]
        args.out_dir = base_out_dir + '/' + test

        # Run ORAC suite
        ou.check_args_cc4cl(args)
        (fileroot, dirs, jid) = ou.cc4cl(args)


except ou.OracError as err:
    cprint('ERROR) ' + err.message, ou.colouring['error'])
except ou.Regression as err:
    print err.message
except KeyboardInterrupt:
    cprint('Execution halted by user.', ou.colouring['error'])
except subprocess.CalledProcessError as err:
    cprint('{:s} failed with error code {:d}. {:s}'.format(
        ' '.join(err.cmd), err.returncode, err.output))

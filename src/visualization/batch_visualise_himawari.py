import re
import os
import tempfile
from datetime import datetime
import glob
import subprocess

import numpy as np
import scipy.ndimage as ndimage

import src.data.readers.load_hrit as load_hrit
import src.config.filepaths as fp



class BatchSystem:
    """Container for syntax to call a batch queuing system.
    Member variables:
    command      - The name of the batch queuing system.
    args         - A dictionary of string format functions, each taking one
                   argument, to produce an argument of the queuing function.
    regex        -  # Regex to parse command output.
    depend_arg   - Command for the job dependencies. This would be an element of
                   args, but has a more complicated structure.
    depend_delim - String require to space consequetive dependencies."""

    def __init__(self, command, regex, depend_arg, depend_delim, args):
        self.command = command
        self.args = args
        self.regex = re.compile(regex)
        self.depend_arg = depend_arg
        self.depend_delim = depend_delim
        self.args.update({'depend' : self.parse_depend})

    def parse_depend(self, item):
        """Deal with slightly more complex syntax for declaring dependencies"""
        if isinstance(item, str):
            return self.depend_arg.format(item)
        else:
            return self.depend_arg.format(self.depend_delim.join(item))

    def print_batch(self, values, exe=None):
        """Returns the queuing shell command. 'exe' is the thing to run."""
        arguments = [self.command]
        arguments.extend([ self.args[key](values[key])
                           for key in values.keys() if values[key] ])
        if type(exe) in [list, tuple]:
            arguments.extend(exe)
        elif exe:
            arguments.append(exe)
        return ' '.join(arguments)

    def parse_out(self, text, key=None):
        """Parse output of queuing function. Returns all regex groups unless
        'key' is specified, where it just returns that."""
        m = self.regex.match(text)
        if m == None:
            raise SyntaxError('Unexpected output from queue system ' + text)
        if key:
            return m.group(key)
        else:
            return m.groupdict()


def proc_params():
    d = {}

    d['start_time'] = datetime(2015, 9, 2, 0, 0)
    d['n_days'] = 15

    d['lat'] = 1.24
    d['lon'] = 103.84

    him_base_path = '/group_workspaces/cems2/nceo_generic/users/xuwd/Himawari8/'
    d['him_base_path'] = him_base_path

    geo_file = os.path.join(him_base_path, 'lcov', 'Himawari_lat_lon.img')
    geostationary_lats, geostationary_lons = load_hrit.geo_read(geo_file)
    zoom = 2  # zoom if using 1km himawara data (B03) for tracking
    d['geostationary_lats'] = ndimage.zoom(geostationary_lats, zoom)
    d['geostationary_lons'] = ndimage.zoom(geostationary_lons, zoom)

    d['output_path'] = '/home/users/dnfisher/data/kcl-fire-aot/him_gif_data'
    return d


def approx_loc_index(pp):
    """
    Get approximate location of a geographic point is an array of
    geographic coordinates
    """

    lat_abs_diff = np.abs(pp['geostationary_lats'] - pp['lat'])
    lon_abs_diff = np.abs(pp['geostationary_lons'] - pp['lon'])
    approx_y, approx_x  = np.unravel_index((lat_abs_diff + lon_abs_diff).argmin(), lat_abs_diff.shape)
    return approx_y, approx_x


def find_segment(approx_y):
    # there are ten 1100 pixel segments in himawari 1 km data
    seg_size = 1100
    min_segment = approx_y / seg_size + 1
    return min_segment


def adjust_y_segment(approx_y, segment):
    seg_size = 1100
    return approx_y - ((segment - 1) * seg_size)


def get_geostationary_fnames(pp, day, image_segment):
    """

    :param plume_time: the time of the MYD observation of the plume
    :param image_segment: the Himawari image segment
    :return: the geostationary files for the day of and the day before the fire
    """
    ym = str(pp['start_time'].year) + str(pp['start_time'].month).zfill(2)
    day = str(int(pp['start_time'].day) + day).zfill(2)

    # get all files in the directory using glob with band 3 for main segment
    p = os.path.join(pp['him_base_path'], ym, day)
    return glob.glob(p + '/*/*/B01/*S' + str(image_segment).zfill(2) + '*')


def submit(pp, geo_file_path, fname, ts, him_segment, approx_y, approx_x):

    # check if time stamp is ok
    if (int(ts.hour) > 11) & (int(ts.hour) < 22):
        return

    # check if we have already processed the file and skip if so
    if os.path.isfile(os.path.join(pp['output_path'], fname.replace('DAT.bz2', 'png'))):
        print os.path.join(pp['output_path'], fname.replace('DAT.bz2', 'png')), 'already processeed'
        return

    # for each file generate a bash script that calls ggf
    (gd, script_file) = tempfile.mkstemp('.sh', 'vis.',
                                         pp['output_path'], True)

    g = os.fdopen(gd, "w")
    g.write('export PYTHONPATH=$PYTHONPATH:/home/users/dnfisher/projects/kcl-fire-aot/\n')
    g.write(fp.path_to_exe_dir + python_exe +
            ' ' + geo_file_path +
            ' ' + fname +
            ' ' + str(him_segment) +
            ' ' + str(approx_y) +
            ' ' + str(approx_x) +
            ' ' + pp['output_path'] + " \n")
    g.write("rm -f " + script_file + "\n")
    g.close()
    os.chmod(script_file, 0o755)

    # generate bsub call using print_batch
    cmd = batch.print_batch(batch_values, exe=script_file)

    # use subprocess to call the print batch command
    try:
        out = subprocess.check_output(cmd.split(' '))
        jid = batch.parse_out(out, 'ID')
    except Exception, e:
        print 'Subprocess failed with error:', str(e)

# setup the batch running class
batch = BatchSystem('bsub',
                   'Job <(?P<ID>\d+)> is submitted to (?P<desc>\w*) queue '
                   '<(?P<queue>[\w\.-]+)>.',
                   '-w "done({})"', ') && done(',
                   {'duration' : '-W {}'.format,
                    'email'    : '-u {}'.format,
                    'err_file' : '-e {}'.format,
                    'job_name' : '-J {}'.format,
                    'log_file' : '-o {}'.format,
                    'order'    : '-R "order[{}]"'.format,
                    'procs'    : '-n {}'.format,
                    'priority' : '-p {}'.format,
                    'queue'    : '-q {}'.format,
                    'ram'      : '-R "rusage[mem={}]"'.format})
batch_values = {'email'    : 'danielfisher0@gmail.com'}


# define python script to run
python_exe = 'himawari_timeseries.py '

# establish himawari segment and approximate location of data point
pp = proc_params()
approx_y, approx_x = approx_loc_index(pp)
him_segment = find_segment(approx_y)
approx_y = adjust_y_segment(approx_y, him_segment)

# iterate over the days
for day in xrange(pp['n_days']):
    geostationary_files_for_day = get_geostationary_fnames(pp, day, him_segment)
    for geo_file_path in geostationary_files_for_day:

        fname = geo_file_path.split('/')[-1]
        ts = datetime.strptime(re.search("[0-9]{8}[_][0-9]{4}", fname).group(),
                               '%Y%m%d_%H%M')

        submit(pp, geo_file_path, fname, ts, him_segment, approx_y, approx_x)


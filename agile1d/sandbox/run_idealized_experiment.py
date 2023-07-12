import argparse
import sys
import os
import importlib.util

from oggm.exceptions import InvalidParamsError
from agile1d.sandbox.define_idealized_experiment import idealized_experiment
from agile1d.sandbox.glaciers_for_idealized_experiments import experiment_glaciers


def parse_args(args):
    """Check input arguments"""

    # CLI args
    description = 'Runs idealized Experiments for agile'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--working_dir', type=str, default=None,
                        help='path to the directory where to write the '
                             'output. Defaults to current directory or '
                             '$OGGM_WORKDIR.')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='path to the directory where to write the '
                             'output. Defaults to current directory or '
                             '$OGGM_OUTDIR.')
    parser.add_argument('--params_file', type=str, default=None,
                        help='path to the OGGM parameter file to use in place '
                             'of the default one.')
    parser.add_argument('--logging_level', type=str, default='WORKFLOW',
                        help='the logging level to use (DEBUG, INFO, WARNING, '
                             'WORKFLOW).')
    parser.add_argument('--experiment_file', type=str,
                        help='path to the experiment file, containing the '
                             'different experiments with their settings')
    parser.add_argument('--print_statistic', nargs='?', const=True,
                        default=False,
                        help='If the idealized statistics should be printed '
                             'out after each run.')

    args = parser.parse_args(args)

    working_dir = args.working_dir
    if not working_dir:
        working_dir = os.environ.get('OGGM_WORKDIR', '')

    output_folder = args.output_folder
    if not output_folder:
        output_folder = os.environ.get('OGGM_OUTDIR', '')

    output_folder = os.path.abspath(output_folder)
    working_dir = os.path.abspath(working_dir)

    # check if information for experiments is correctly given in separate file
    spec = importlib.util.spec_from_file_location("experiments",
                                                  args.experiment_file)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    if not hasattr(foo, 'inversion_settings_all'):
        raise InvalidParamsError('No variable "inversion_settings_all" in '
                                 'provided experiments file!')
    if not hasattr(foo, 'use_experiment_glaciers'):
        raise InvalidParamsError('No variable "experiment_glaciers" in '
                                 'provided experiments file!')
    if not hasattr(foo, 'inversion_settings_individual'):
        raise InvalidParamsError('No variable "inversion_settings_individual" in '
                                 'provided experiments file!')
    use_experiment_glaciers = foo.use_experiment_glaciers
    for glacier in use_experiment_glaciers:
        if glacier not in experiment_glaciers.keys():
            raise InvalidParamsError(f'{glacier} not supported! First must be '
                                     'added to example glaciers!')

    return dict(working_dir=working_dir, output_folder=output_folder,
                params_file=args.params_file,
                logging_level=args.logging_level,
                inversion_settings_all=foo.inversion_settings_all,
                inversion_settings_individual=foo.inversion_settings_individual,
                use_experiment_glaciers=use_experiment_glaciers,
                print_statistic=args.print_statistic
                )


def main():
    """Script entry point"""

    idealized_experiment(**parse_args(sys.argv[1:]))

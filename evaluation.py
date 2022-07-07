from lib.process.actions import plot_sample_figs
from lib.utils.arguments import process_command_line


def main(args):
    print(f'Runing evaluation: \n {args}')
    kwargs = dict(args._get_kwargs())
    for action in args.actions:
        if action == 'plot_samples':
            plot_sample_figs(_sample_to_plot= args.sample_to_plot, **kwargs)
        else:
            print(f'Action {action} not implemented, options are: plot_samples')


if __name__ == '__main__':
    args = process_command_line()
    main(args)

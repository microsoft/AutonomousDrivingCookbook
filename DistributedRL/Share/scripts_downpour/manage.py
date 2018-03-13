import os
import sys
import argparse

def setup_logs():
    output_dir = 'Z:\\logs\\{0}\\trainer'.format(os.environ['experiment_name'])
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    sys.stdout = open(os.path.join(output_dir, '{0}.stdout.txt'.format(os.environ['AZ_BATCH_NODE_ID'])), 'w')
    sys.stderr = open(os.path.join(output_dir, '{0}.stderr.txt'.format(os.environ['AZ_BATCH_NODE_ID'])), 'w')



if __name__ == "__main__":
    print('IN MANAGE.PY')
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "downpour.settings")
    
    custom_args = sys.argv[3:]
    original_args = sys.argv[:3]
    #known_args = ['data_dir', 'role', 'experiment_name', 'batch_update_frequency']
    parser = argparse.ArgumentParser(add_help=False)
    for arg in custom_args:
        arg_name = arg.split('=')[0]
        parser.add_argument(arg_name)
    args, _ = parser.parse_known_args(custom_args)
    args = vars(args)
    for arg in args:
        os.environ[arg] = args[arg].split('=')[1]
    
    print('**************')
    print('OS.ENVIRON')
    print(os.environ)
    print('**************')
    
    setup_logs()
    
    print('MANAGE.PY: name: {0}'.format(__name__))
    print('TO STDERR: name: {0}'.format(__name__), file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(original_args)

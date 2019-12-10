import os
import os.path as osp
import yaml
import time
import argparse
import shutil
from easydict import EasyDict as edict

def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Running Experiments of Deep Graph Generation")
  parser.add_argument(
      '-c',
      '--config_file',
      type=str,
      default="config/gran_grid.yaml",
      required=True,
      help="Path of config file")
  parser.add_argument(
      '-l',
      '--log_level',
      type=str,
      default='INFO',
      help="Logging Level, \
        DEBUG, \
        INFO, \
        WARNING, \
        ERROR, \
        CRITICAL")
  parser.add_argument('-m', '--comment', help="Experiment comment")
  parser.add_argument('-t', '--test', help="Test model", action='store_true')
  args = parser.parse_args()

  return args


def get_config(config_file, exp_dir=None, is_test=False):
  """ Construct and snapshot hyper parameters """
  config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
  # config = edict(yaml.load(open(config_file, 'r')))

  if config.runner == 'GranRunner':
    print('use GranRunner, assert block size and stride and fwd pass')
    assert config.model.block_size == 1
    assert config.model.sample_stride == 1
    assert config.dataset.num_fwd_pass == 1

  # create hyper parameters
  config.run_id = str(os.getpid())
  config.exp_name = '_'.join([
      config.exp_name_prefix, config.model.name, config.dataset.name,
      time.strftime('%Y-%b-%d-%H-%M-%S'), config.run_id
  ])
  if is_test:
    config.exp_name = 'test_' + config.exp_name

  if exp_dir is not None:
    config.exp_dir = exp_dir
  
  if config.train.is_resume and not is_test:
    config.save_dir = config.train.resume_dir
    save_name = os.path.join(config.save_dir, 'config_resume_{}.yaml'.format(config.run_id))  
  else:    
    config.save_dir = os.path.join(config.exp_dir, config.exp_name)
    save_name = os.path.join(config.save_dir, 'config.yaml')

  # snapshot hyperparameters
  mkdir(config.exp_dir)
  mkdir(config.save_dir)

  # -------------------- code copy --------------------
  if not config.train.is_resume or is_test:
    # TODO: find better approach
    repo_basename = osp.basename(osp.dirname(osp.dirname(osp.abspath(__file__))))
    repo_path = osp.join(config.save_dir, repo_basename)
    os.makedirs(repo_path, mode=0o777, exist_ok=True)

    walk_res = os.walk('.')
    useful_paths = [path for path in walk_res if
                    '.git' not in path[0] and
                    'data' not in path[0] and
                    'exp' not in path[0] and
                    'configs' not in path[0] and
                    '__pycache__' not in path[0] and
                    'tee_dir' not in path[0] and
                    'tmp' not in path[0]]
    # print('useful_paths', useful_paths)
    for p in useful_paths:
      for item in p[-1]:
        if not (item.endswith('.py') or item.endswith('.cpp') or item.endswith('.h') or item.endswith('.md')):
          continue
        old_path = osp.join(p[0], item)
        new_path = osp.join(repo_path, p[0][2:], item)
        basedir = osp.dirname(new_path)
        os.makedirs(basedir, mode=0o777, exist_ok=True)
        shutil.copyfile(old_path, new_path)
    # If cannot find file, will raise FileNotFoundError
    # The destination location must be writable; otherwise, an OSError exception will be raised.
    #  If dst already exists, it will be replaced. Special files such as character or block devices
    #  and pipes cannot be copied with this function.

  yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

  return config


def edict2dict(edict_obj):
  dict_obj = {}

  for key, vals in edict_obj.items():
    if isinstance(vals, edict):
      dict_obj[key] = edict2dict(vals)
    else:
      dict_obj[key] = vals

  return dict_obj


def mkdir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)

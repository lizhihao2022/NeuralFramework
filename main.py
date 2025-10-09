from config import parser
from utils import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config
from trainers import base_procedure


def main():
    args = parser.parse_args()
    args = vars(args)
    args = load_config(args)

    saving_path, saving_name = set_up_logger(args)

    args['train']['saving_path'] = saving_path
    args['train']['saving_name'] = saving_name
    save_config(args, saving_path)
    set_device(args['cuda'], args['device'])
    set_seed(args['random_seed'])

    if args['data']['dataset'] == 'Base':
        base_procedure(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()

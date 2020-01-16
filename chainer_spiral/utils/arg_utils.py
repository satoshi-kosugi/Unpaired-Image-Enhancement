import json
import logging


def load_args(args, arg_log, exceptions=None):
    """ load arguments from arg_log """
    with open(arg_log, 'r') as f:
        loaded_args = json.load(f)

    # overwrite args by loded_args
    for key, value in loaded_args.items():
        if exceptions:
            if not key in exceptions:
                setattr(args, key, value)
        else:
            setattr(args, key, value)

    return args


def print_args(args, logger=None):
    """ print argument list by using logger """
    logger = logger or logging.getLogger(__name__)
    logger.warning('================================')
    for key, value in vars(args).items():
        logger.warning('%s : %s', key, value)
    logger.warning('================================')

import logging
import logging.config


def setup_TAS_logger():
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                # 'format': '%(levelname)s: %(message)s'
                'format': '[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s.%(msecs)03d: %(message)s',
                'datefmt': '%Y-%m-%dT%H:%M:%S%z'
            }
        },
        'handlers': {
            'stdout': {
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'root': {
                'level': 'INFO',
                'handlers': ['stdout']
            },
            'matplotlib': {
                'level': 'INFO'
            }
        }
    }

    logging.config.dictConfig(logging_config)
    return logging.getLogger('TAS_logger')
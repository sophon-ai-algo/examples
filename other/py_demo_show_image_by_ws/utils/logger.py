import os
import psutil
import logging
import logging.config


# 创建日志目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            'format': '%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
        },
        'standard': {
            'format': '%(asctime)s [%(threadName)s:%(thread)d] [%(name)s:%(lineno)d] [%(levelname)s]- %(message)s'
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": 'DEBUG',
            "formatter": "simple",
        },

        "default": {
            'class': 'logging.handlers.RotatingFileHandler',
            "level": 'DEBUG',
            "formatter": "simple",
            "filename": os.path.join(LOG_DIR, 'manager.log'),
            'maxBytes': 1024 * 1024 * 5,
            "backupCount": 5,
            "encoding": "utf8"
        },
    },

    "root": {
        'handlers': ['default', 'console'],
        'level': 'DEBUG',
        'propagate': False
    }
}

logging.config.dictConfig(LOGGING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("tornado.access").setLevel(logging.WARNING)


def get_logger(file):
    log = logging.getLogger(file)
    return log

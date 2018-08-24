#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:37:18 2018

@author: jack.lingheng.meng
"""
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='example3.log',level=logging.DEBUG)
# assuming loglevel is bound to the string value obtained from the
# command line argument. Convert to upper case to allow the user to
# specify --log=DEBUG or --log=debug
#numeric_level = getattr(logging, loglevel.upper(), None)
#if not isinstance(numeric_level, int):
#    raise ValueError('Invalid log level: %s' % loglevel)
#logging.basicConfig(level=numeric_level)

logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
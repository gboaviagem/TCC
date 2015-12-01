import logging
LOG_FILENAME = 'log.txt'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG,)

# -----------------------
try:
    print 2/0
except Exception, e:
    logging.error(e, exc_info=True)
# -----------------------
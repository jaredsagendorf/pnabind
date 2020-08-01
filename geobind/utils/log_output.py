# builtin modules
import logging

def logOutput(output, logger):
    for line in output:
        logger(line)

"""Gunicorn settings for sentence-api (production server).

    gunicorn -c gunicorn.conf.py sentence-api:app

The address comes from the same variables as the development server. Override
any other setting with GUNICORN_CMD_ARGS (e.g. "--threads 8"), which takes
precedence over this file.
"""

import os

bind = f"{os.environ.get('EMBEDDING_HOST', '0.0.0.0')}:{os.environ.get('EMBEDDING_PORT', '8000')}"

# One process: every worker would load its own copy of the models. Concurrency
# comes from threads; torch releases the GIL while it computes, and the model
# caches are guarded by a lock.
workers = 1
worker_class = "gthread"
threads = 4
# The models load in a thread started when the app is imported; a preloaded app
# would start it in the master, and threads do not survive the fork.
preload_app = False

# A cold load of a newly allowed model downloads it; give it time.
timeout = 120
graceful_timeout = 30

# No runtime control socket: it would live in the home of a user that has none.
control_socket_disable = True

accesslog = "-"
errorlog = "-"

"""Project package initialization for database adapters."""

import pymysql

# Allow Django's MySQL backend to use PyMySQL as MySQLdb.
pymysql.install_as_MySQLdb()

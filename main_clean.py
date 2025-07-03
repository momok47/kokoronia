# Main Application
# Original size: 6239 bytes
# 重要なアプリケーションロジックが含まれています

import os
import sys
from django.core.wsgi import get_wsgi_application

# Django設定
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
application = get_wsgi_application()

if __name__ == '__main__':
    print('Lounge Application Starting...')
    # メインアプリケーションロジックをここに追加


#!/bin/bash
# KOKORONIA 環境設定スクリプト

echo "🚀 KOKORONIA 環境を設定中..."

# プロジェクトディレクトリに移動
cd "$(dirname "$0")"

# .envファイルから環境変数を読み込み
if [ -f .env ]; then
    echo "📁 .envファイルを読み込み中..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ .env設定完了"
else
    echo "⚠️  .envファイルが見つかりません"
fi

# 仮想環境をアクティベート
if [ -d .venv ]; then
    echo "🐍 仮想環境をアクティベート中..."
    source .venv/bin/activate
    echo "✅ 仮想環境アクティベート完了"
else
    echo "⚠️  .venvディレクトリが見つかりません"
fi

# GCS設定の確認
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        echo "✅ GCS認証: $GOOGLE_APPLICATION_CREDENTIALS"
    else
        echo "❌ GCS認証ファイルが見つかりません: $GOOGLE_APPLICATION_CREDENTIALS"
    fi
else
    echo "❌ GOOGLE_APPLICATION_CREDENTIALS が設定されていません"
fi

# バケット名確認
if [ -n "$GCS_BUCKET_NAME" ]; then
    echo "✅ GCSバケット: $GCS_BUCKET_NAME"
else
    echo "❌ GCS_BUCKET_NAME が設定されていません"
fi

echo ""
echo "🎯 KOKORONIA環境設定完了！"
echo "次のコマンドでWebサーバーを起動できます:"
echo "  cd src/webapp && python manage.py runserver"
echo "" 
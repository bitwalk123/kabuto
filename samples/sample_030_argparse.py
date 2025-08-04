import argparse


def main():
    # 1. パーサーを作成
    parser = argparse.ArgumentParser(description='このプログラムは指定されたファイルを処理します。')

    # 2. 位置引数を追加
    # 'file_path' は必須の引数で、コマンドライン上で位置によって指定されます。
    parser.add_argument('file_path', type=str, help='処理するファイルのパス')

    # 3. オプション引数を追加
    # '-o' または '--output' で指定するオプション引数。
    # 'dest' で引数を取得する際の変数名を指定できます。
    # 'default' で引数が指定されなかった場合のデフォルト値を設定できます。
    parser.add_argument('-o', '--output', dest='output_dir', type=str, default='./',
                        help='出力ファイルを保存するディレクトリ (デフォルト: ./)')

    # 4. ブール値のフラグを追加
    # 'action="store_true"' を使うと、フラグが指定された場合に True、そうでなければ False になります。
    parser.add_argument('--verbose', action='store_true',
                        help='詳細なログを出力します')

    # 5. 引数をパース
    args = parser.parse_args()

    # 6. パースした引数を使用
    print(f'処理対象ファイル: {args.file_path}')
    print(f'出力ディレクトリ: {args.output_dir}')
    if args.verbose:
        print('詳細モードが有効です。')


if __name__ == '__main__':
    main()

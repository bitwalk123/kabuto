import argparse


def main():
    # 1. パーサーを作成
    parser = argparse.ArgumentParser(
        description='このプログラムは指定されたオプションを処理します。'
    )

    # 2. 位置引数を追加
    # 'file_path' は必須の引数で、コマンドライン上で位置によって指定されます。
    # parser.add_argument('file_path', type=str, help='処理するファイルのパス')

    # 3. オプション引数を追加
    parser.add_argument('-xl', '--excel',
                        dest='path_excel',
                        type=str,
                        default='./targets.xlsm',
                        help='RSS用のExcelファイル (デフォルト: ./targets.xlsm)')

    # 4. ブール値のフラグを追加
    # 'action="store_true"' を使うと、フラグが指定された場合に True、そうでなければ False になります。
    parser.add_argument("-d", "--debug",
                        action='store_true',
                        help="デバッグモードで起動")

    # 5. 引数をパース
    args = parser.parse_args()

    # 6. パースした引数を使用
    print(f'RSS用Excelファイル: {args.path_excel}')

    if args.debug:
        print('デバッグモードが有効です。')


if __name__ == '__main__':
    main()

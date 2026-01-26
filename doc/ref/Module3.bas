Attribute VB_Name = "Module3"

'現物取引で楽天225ダブルブルを1株買って、その株を売却した場合のコード
'Range("L11")を空白にしておいたほうが解りやすい。エラーの内容を教えてくれるから
'エラー[マーケットスピードII RSSの利用に関する確認書兼同意書をご覧ください。ウェブログイン後、マイメニューの「お取引に関わる重要書面」よ]
'上記エラーはWebの確認書兼同意書にチェックしていないからです
'執行条件L27=3寄付で予約しています。予約出来たら成功です。寄付は朝一番と午後一番とあります

'現物買い
Sub test_3()

'リセット用
'Range("K11").Formula = "=IF(L19=0,RssStockOrder(L13,L14,L15,L20,L21,L22,L24,L25,L26,L27,L28,L29,L30,L31,L32,L33,L34,L36,L37,L38),IF(L19=1,RssMarginOpenOrder(L13,L14,L15,L20,L21,L22,L23,L24,L25,L26,L27,L28,L29,L30,L31,L32,L33,L34,L35,L36,L37,L38),""注文種類を選んでください""))"
'Exit Sub

Range("J9").Value = 1458    '①楽天225ダブルブル

Call CommandButton11_Click  '②発注IDセットボタン

Range("L14").Value = 0                  '発注トリガー
Range("L15").Value = Range("J9").Value  '銘柄コード

                  '③L列に入力
Range("L19").Value = 0    '0:現物取引  1:信用取引
Range("L20").Value = 3    '1:売り　3:買い
Range("L21").Value = 0
Range("L22").Value = 0
Range("L24").Value = 1     '注文数量
Range("L25").Value = 0     '0:成行
Range("L27").Value = 3     '1:本日中  3:寄付
Range("L29").Value = 0
Range("L34").Value = 0

'IF(L19=0,RssStockOrder(L13,L14,の関数がセットされる

Call CommandButton12_Click  '④発注ボタン

Call CommandButton11_Click  'リセット


End Sub


'現物売り
Sub test_1()

Range("J9").Value = 1458

Call CommandButton11_Click

Range("L14").Value = 0                  '発注トリガー
Range("L15").Value = Range("J9").Value  '銘柄コード

Range("L19").Value = 0
Range("L20").Value = 1    '売る
Range("L21").Value = 0
Range("L22").Value = 0
Range("L24").Value = 1
Range("L25").Value = 0
Range("L27").Value = 3
Range("L29").Value = 0
Range("L34").Value = 0

'IF(L19=0,RssStockOrder(L13,L14,の関数がセットされる

Call CommandButton12_Click

Call CommandButton11_Click

End Sub

'信用取引で楽天225ダブルブルを1株買って、その株を返済売りした場合のコード
'Range("L11")を空白にしておいたほうが解りやすい。エラーの内容を教えてくれるから

'信用新規買い
Sub shinyo_3()

Range("J9").Value = 1458    '①楽天225ダブルブル

Call CommandButton11_Click  '②発注IDセットボタン

                  '③L列に入力
                  
Range("L14").Value = 0                  '発注トリガー
Range("L15").Value = Range("J9").Value  '銘柄コード
                  
Range("L19").Value = 1    '0:現物取引  1:信用取引
Range("L20").Value = 3    '1:売り　3:買い
Range("L21").Value = 0
Range("L22").Value = 0
Range("L23").Value = 1     '信用区分　1:制度信用(6ヶ月)
Range("L24").Value = 1     '注文数量
Range("L25").Value = 0     '0:成行
Range("L27").Value = 3     '1:本日中  3:寄付
Range("L29").Value = 0
Range("L34").Value = 0

'IF(L19=1,RssMarginOpenOrder(L13,L14 の関数がセットされる

Call CommandButton12_Click    '④発注ボタン

Call CommandButton11_Click    'リセット

End Sub


'信用返済売り
Sub hensai_1()

Range("J9").Value = 1458    '①楽天225ダブルブル

Call CommandButton11_Click  '②発注IDセットボタン

                  '③L列に入力
                  
Range("L14").Value = 0                  '発注トリガー
Range("L15").Value = Range("J9").Value  '銘柄コード
                  
Range("L19").Value = 1    '0:現物取引  1:信用取引
Range("L20").Value = 1    '1:売り　3:買い
Range("L21").Value = 0
Range("L22").Value = 0
Range("L23").Value = 1     '信用区分　1:制度信用(6ヶ月)
Range("L24").Value = 1     '注文数量
Range("L25").Value = 0     '0:成行
Range("L27").Value = 3     '1:本日中  3:寄付
Range("L29").Value = 0
Range("L34").Value = 0

     Range("K39").Value = "20220610"  '建日
     Range("L39").Value = 15450       '建単価
     Range("M39").Value = 1           '建市場　 1:東証



'返済売は買建玉を売り渡す取引、返済買は売建玉を買戻す取引です。
'K11セルに信用返済関数を入力セットする
Range("K11").Formula = "= RssMarginCloseOrder(L13,L14,L15,L20,L21,L22,L23,L24,L25,L26,L27,L28,L29,K39,L39,M39,L30,L31,L32,L33)"

Call CommandButton12_Click  '④発注ボタン

Call CommandButton11_Click  'リセット

'元の状態に戻しておく
Range("K11").Formula = "=IF(L19=0,RssStockOrder(L13,L14,L15,L20,L21,L22,L24,L25,L26,L27,L28,L29,L30,L31,L32,L33,L34,L36,L37,L38),IF(L19=1,RssMarginOpenOrder(L13,L14,L15,L20,L21,L22,L23,L24,L25,L26,L27,L28,L29,L30,L31,L32,L33,L34,L35,L36,L37,L38),""注文種類を選んでください""))"

End Sub

'信用新規売建1　、信用返済買3　のボタンはセットした
'Sub shinyo_1()   Sub hensai_3() を作成してマクロ登録して下さい



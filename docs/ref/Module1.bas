Attribute VB_Name = "Module1"
Private ha_id As Long

Sub SET_HA_ID(sheet_id As String)
' 発注IDセット　＆　トリガーOFF
    If ha_id = 0 Then
        ha_id = 10001
    End If

    If ha_id > 20000 Then
        MsgBox ("発注IDが2000を超えています。一度ファイルを閉じてください。")
    End If
    
    '既に使用されている最大番号 +1 する
    If ha_id <= Sheets("発注ID一覧").Range("A4") Then
        ha_id = Sheets("発注ID一覧").Range("A4") + 1
    End If

    Select Case sheet_id
    Case "1"
      ActiveSheet.Range("L14").Value = 0
      ActiveSheet.Range("L13").Value = ha_id
          Case "2"
      ActiveSheet.Range("N14").Value = 0
      ActiveSheet.Range("N13").Value = ha_id
    Case "3"
      ActiveSheet.Range("L20").Value = 0
      ActiveSheet.Range("L19").Value = ha_id
    Case "4"
      ActiveSheet.Range("D9").Value = 0
      ActiveSheet.Range("C13").Value = ha_id
    Case "5"
      ActiveSheet.Range("C9").Value = 0
      ActiveSheet.Range("C13").Value = ha_id
    Case "6"
      ActiveSheet.Range("R19").Value = 0
      ActiveSheet.Range("O22").Value = ha_id
    End Select

    ha_id = ha_id + 1
    
End Sub

Sub SET_TRIGGER(sheet_id As String)
'トリガーON
    Select Case sheet_id
    Case "1"
      ActiveSheet.Range("L14").Value = 1
    Case "2"
      ActiveSheet.Range("N14").Value = 1
    Case "3"
      ActiveSheet.Range("L20").Value = 1
    Case "4"
      ActiveSheet.Range("D9").Value = 1
    Case "5"
      ActiveSheet.Range("C9").Value = 1
    Case "6"
      ActiveSheet.Range("R19").Value = 1
    End Select
        
End Sub




import logging
import subprocess
from pathlib import Path
from typing import Generator

from PySide6.QtCore import QObject, Signal

from structs.res import AppRes


class UploadWorker(QObject):
    """
    設定ファイルを指定したサーバー、場所にアップロードして保存する
    """
    finished = Signal()

    def __init__(self, res: AppRes, files: Generator):
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        super().__init__()
        self.res = res
        self.files = files

    def run(self):
        key = str(Path(self.res.ssh_key_path).expanduser())
        target = f"{self.res.remote_user}@{self.res.remote_host}:{self.res.remote_conf_dir}"

        for f in self.files:
            cmd = ["scp", "-i", key, str(f), target]
            self.logger.info(f"{__name__}: Uploading {f.name} ...")
            subprocess.run(cmd)

        self.finished.emit()

from funcs.logs import setup_logging
from modules.radcliffe import Radcliffe

if __name__ == "__main__":
    main_logger = setup_logging()
    app = Radcliffe()
    app.run()
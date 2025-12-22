from funcs.logs import setup_logging
from modules.apostle import Apostle

if __name__ == "__main__":
    main_logger = setup_logging()
    app = Apostle()
    app.run()
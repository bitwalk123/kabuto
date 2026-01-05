from funcs.logs import setup_logging
from modules.disciple import Disciple

if __name__ == "__main__":
    main_logger = setup_logging()
    app = Disciple()
    app.run()
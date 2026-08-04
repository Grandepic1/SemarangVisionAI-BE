import os

from dotenv import load_dotenv
from sqlalchemy.engine import URL

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Bandung Vision AI")

# --- Database settings (component-based; no full URL in env) ---
DB_TYPE = os.getenv("DB_TYPE", "postgresql").lower()  # "postgresql" or "mysql"
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "")
DB_USER = os.getenv("DB_USER", "root" if DB_TYPE == "mysql" else "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "bandungvision")

_DRIVER_BY_TYPE = {
    "postgresql": ("postgresql+psycopg2", "5432"),
    "mysql": ("mysql+pymysql", "3306"),
}


def build_database_url() -> str:
    """Build a SQLAlchemy URL from the individual DB settings in env.

    URL.create() handles escaping of special characters (e.g. "@" or ":"
    inside the password) correctly, unlike manual string formatting.
    """
    if DB_TYPE not in _DRIVER_BY_TYPE:
        supported = ", ".join(_DRIVER_BY_TYPE)
        raise ValueError(f"Unsupported DB_TYPE '{DB_TYPE}'. Supported types: {supported}")

    driver, default_port = _DRIVER_BY_TYPE[DB_TYPE]
    port = int(DB_PORT or default_port)
    return URL.create(
        drivername=driver,
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=port,
        database=DB_NAME,
    ).render_as_string(hide_password=False)


DATABASE_URL = build_database_url()

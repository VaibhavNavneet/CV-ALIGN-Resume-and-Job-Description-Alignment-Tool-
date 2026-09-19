"""Create the PostgreSQL run-history table before deploying."""

from multi_agent_resume_screener.settings import get_settings
from multi_agent_resume_screener.storage.runs import PostgresRunStore


def main() -> None:
    url = get_settings().database_url
    if not url:
        raise SystemExit("Set DATABASE_URL in the environment or .env first.")
    PostgresRunStore(url).initialize()
    print("Run-history schema is ready.")


if __name__ == "__main__":
    main()

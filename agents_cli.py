"""Entry point for the multi-agent demonstration CLI.

    ./.venv/bin/python agents_cli.py demo            # one live case, streamed agent by agent
    ./.venv/bin/python agents_cli.py compare         # detector vs single agent vs MAS
    ./.venv/bin/python agents_cli.py live --limit 8  # batch over the mock environment
"""

from zeroday_verify.agents.cli import main

if __name__ == "__main__":
    main()

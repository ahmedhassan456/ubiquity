"""Lets the CLI run as `python -m ubiquity`, without the console script."""

from .cli.main import main

raise SystemExit(main())

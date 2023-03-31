import logging

import click
import structlog


@click.group()
@click.option(
    "-d",
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging",
)
@click.option(
    "--log-format",
    type=click.Choice(["json", "human"]),
    default="human",
    help="Log format",
)
def cli(debug: bool, log_format: str) -> None:
    processors: list[structlog.types.Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.PATHNAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    elif log_format == "human":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        raise ValueError("Invalid log format")

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    loglevel = logging.DEBUG if debug else logging.INFO
    LOG_FORMAT = "%(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    structlog.get_logger("chatgpt_playground").setLevel(level=loglevel)

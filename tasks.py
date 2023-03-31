from concurrent.futures import ThreadPoolExecutor, as_completed

from invoke import task


@task
def black_format(ctx):
    """Format code with black."""
    ctx.run("poetry run black .")


@task
def isort_format(ctx):
    """Format code with isort."""
    ctx.run("poetry run isort .")


def parallel_run(ctx, tasks):
    """Run the tasks in parallel."""
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(task, ctx) for task in tasks]
        for future in as_completed(futures):
            print(future.result())


@task
def format(ctx):
    """Format code."""
    tasks = [
        black_format,
        isort_format,
    ]
    parallel_run(ctx, tasks)


@task
def black_check(ctx):
    """Check code formatting with black."""
    ctx.run("poetry run black --check .")


@task
def isort_check(ctx):
    """Check code formatting with isort."""
    ctx.run("poetry run isort --check-only .")


@task
def mypy_check(ctx):
    """Check code formatting with mypy."""
    ctx.run("poetry run mypy .")


@task
def flake8_check(ctx):
    """Check code formatting with flake8."""
    ctx.run("poetry run flake8 .")


@task
def lint(ctx):
    """Lint code."""
    tasks = [
        black_check,
        isort_check,
        mypy_check,
        flake8_check,
    ]
    parallel_run(ctx, tasks)

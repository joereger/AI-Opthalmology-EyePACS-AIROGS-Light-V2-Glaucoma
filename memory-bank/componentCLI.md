# Component: CLI & Orchestration

## Description

This component acts as the central entry point and orchestrator for the entire application. It provides a command-line interface (CLI) that allows users to execute individual pipeline phases, manages global configuration loading, and sets up system-wide services like logging.

It effectively decouples the user interaction from the underlying business logic of each phase, dynamically dispatching commands to the appropriate module based on user input.

## Key Files

*   `main.py`
    *   **Role**: The entry script. It invokes the `src/cli.py` logic.
*   `src/cli.py`
    *   **Role**: The core CLI implementation using the `click` library. It defines the commands, parses arguments, configures logging, and dynamically imports/executes phase modules.
*   `src/config_loader.py`
    *   **Role**: Responsible for loading and merging configuration files (`config.yaml`), handling the precedence of global vs. phase-specific settings.
*   `src/utils/logger.py`
    *   **Role**: Configures the Python logging facility based on `logger_config.yaml`.

## Technology and Architecture

*   **Framework**: `click` is used for creating the CLI. It provides decorators for defining commands and options, and handles help generation automatically.
*   **Dynamic Import**: To maintain modularity and startup performance, phase modules are imported only when their specific command is invoked.
*   **Configuration**: Uses `PyYAML` to parse `config.yaml`. The `config_loader` typically merges a global config with phase-specific configs.

## TODO

*   No specific TODOs listed in active context.

## Progress

*   **2025-05-08**: Integrated Phase 07 (Activation Maximization) into the CLI (`src/cli.py`), allowing it to be triggered via `python main.py phase07`.
*   **Previous**: Initial implementation of CLI structure and integration of Phases 01-06.

## Key Lessons and Insights

*   Dynamic importing of phase modules prevents circular dependencies and reduces startup time, especially as the number of phases grows.

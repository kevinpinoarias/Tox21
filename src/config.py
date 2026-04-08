from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, default="config.yaml"
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the config file is empty or invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(config, dict):
        raise ValueError("Config must parse into a dictionary.")

    return config


def get_required(config: Dict[str, Any], *keys: str) -> Any:
    """
    Retrieve a nested required config value.

    Example
    -------
    get_required(config, "data", "path")

    Parameters
    ----------
    config : dict
        Full config dictionary.
    *keys : str
        Nested key path.

    Returns
    -------
    Any
        Config value.

    Raises
    ------
    KeyError
        If any key is missing.
    """
    current = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            full_path = " -> ".join(keys)
            raise KeyError(f"Missing required config key: {full_path}")
        current = current[key]

    return current


def get_optional(
    config: Dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Retrieve a nested optional config value.

    Example
    -------
    get_optional(config, "split", "test_size", default=0.2)

    Parameters
    ----------
    config : dict
        Full config dictionary.
    *keys : str
        Nested key path.
    default : Any, optional
        Default value if key path does not exist.

    Returns
    -------
    Any
        Config value or default.
    """
    current = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def validate_config(config: Dict[str, Any]) -> None:
    """
    Perform minimal validation of required project config structure.

    Parameters
    ----------
    config : dict
        Parsed config dictionary.

    Raises
    ------
    KeyError
        If required sections/keys are missing.
    ValueError
        If unsupported values are found.
    """
    # Required top-level sections
    required_sections = ["project", "data", "split", "featurization", "model", "outputs"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required config section: '{section}'")

    # Required nested keys
    get_required(config, "data", "path")
    get_required(config, "featurization", "method")
    get_required(config, "model", "name")
    get_required(config, "outputs", "base_dir")

    # Validate supported feature methods
    featurization_method = get_required(config, "featurization", "method")
    valid_featurizers = {"morgan"}
    if featurization_method not in valid_featurizers:
        raise ValueError(
            f"Unsupported featurization method '{featurization_method}'. "
            f"Supported: {sorted(valid_featurizers)}"
        )

    # Validate supported model names
    model_name = get_required(config, "model", "name")
    valid_models = {"logistic_regression", "random_forest", "xgboost", "neural_network"}
    if model_name not in valid_models:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Supported: {sorted(valid_models)}"
        )


def load_and_validate_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration in one step.

    Parameters
    ----------
    config_path : str or Path, default="config.yaml"
        Path to config file.

    Returns
    -------
    dict
        Validated config dictionary.
    """
    config = load_config(config_path)
    validate_config(config)
    return config
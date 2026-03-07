import sys
import os
from unittest.mock import patch

# Add src and src/core to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/core"))
)


@patch("flwr.server.start_server")
def test_main_starts_server(mock_start_server):
    # Import server inside the test to allow patching of flwr before it runs
    import core.server as server

    # Execute main
    server.main()

    # Check if start_server was called
    mock_start_server.assert_called_once()

    # Check arguments
    call_args, call_kwargs = mock_start_server.call_args
    assert "server_address" in call_kwargs
    assert call_kwargs["server_address"] == "0.0.0.0:8080"

    # Check if strategy was passed correctly
    strategy = call_kwargs["strategy"]
    assert strategy is not None
    assert strategy.fraction_fit == 1.0

    # Check config functions
    fit_config_fn = strategy.on_fit_config_fn
    eval_config_fn = strategy.on_evaluate_config_fn

    fit_cfg = fit_config_fn(server_round=5)
    assert fit_cfg["server_round"] == 5

    eval_cfg = eval_config_fn(server_round=2)
    assert eval_cfg["server_round"] == 2
    assert "eval_episodes" in eval_cfg

import os
import pandas as pd
import pytest

RESULTS_FILE = "NFR_Testing/Performance/results.jtl"

def parse_csv_jtl(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    df = pd.read_csv(filepath)
    total = len(df)

    failures = len(df[df["success"] == False])

    error_rate = (failures / total * 100) if total > 0 else 0

    avg_response_time = df["elapsed"].mean() if total > 0 else 0

    return error_rate, avg_response_time

@pytest.mark.parametrize("filepath", [RESULTS_FILE])
def test_jmeter_csv_results(filepath):
    error_rate, avg_response_time = parse_csv_jtl(filepath)

    assert error_rate == 0, f"Error rate is {error_rate}%, expected 0%."
    assert avg_response_time < 500, f"Average response time is {avg_response_time}ms, expected less than 500ms."
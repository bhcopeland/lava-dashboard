import pandas as pd
import json

def load_and_process_data(file_path):
    """
    Loads and processes LAVA job data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        tuple: A tuple containing two pandas.DataFrames: (jobs, events).
    """
    with open(file_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and 'jobs' in raw and 'events' in raw:
        jobs = pd.json_normalize(raw['jobs'])
        events = pd.json_normalize(raw['events'])
        events['timestamp'] = pd.to_datetime(events['timestamp'], format='mixed')
    else:
        # Fallback for the old format
        jobs = pd.json_normalize(raw)
        events = pd.DataFrame(columns=['timestamp', 'queue_length'])

    # Remove all "qemu" devices
    jobs = jobs[~jobs['requested_device_type'].str.startswith('qemu')]

    # Parse dates
    for column in ('submit_time', 'start_time', 'end_time'):
        jobs[column] = pd.to_datetime(jobs[column], format='mixed')

    # Discard duplicated id values (retaining the most recent record)
    jobs.drop_duplicates(subset=['id'], keep='last', inplace=True)
    
    jobs.sort_values(by=['submit_time'], inplace=True)

    # Simple synthesis
    jobs['latency'] = jobs['start_time'] - jobs['submit_time']
    jobs['latency_in_minutes'] = jobs['latency'].dt.total_seconds() / 60
    jobs['execution_time'] = jobs['end_time'] - jobs['start_time']

    return jobs, events
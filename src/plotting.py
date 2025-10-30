import os
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import math

def plot_submitter_breakdown(jobs, output_path):
    """
    Generates a pie chart showing the breakdown of job submitters.

    Args:
        jobs (pandas.DataFrame): The DataFrame containing the job data.
        output_path (str): The path to save the HTML file.
    """
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()

    count_data = pd.DataFrame(jobs[(jobs.submit_time > start_time) & (jobs.submit_time < end_time)].groupby('submitter').size(), columns=['count'])

    labels = count_data.index
    values = count_data['count']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title_text='Breakdown of submitters')

    html_file_path = os.path.join(output_path, "breakdown_submitters.html")
    fig.write_html(html_file_path)

def plot_execution_time_by_submitter(jobs, output_path):
    """
    Generates a table showing the execution time and job count by submitter.

    Args:
        jobs (pandas.DataFrame): The DataFrame containing the job data.
        output_path (str): The path to save the HTML file.
    """
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.start_time > start_time) & (jobs.start_time < end_time) & (pd.notnull(jobs.end_time))]
    slice = pd.DataFrame(window[['submitter', 'execution_time']])
    slice['number_of_jobs'] = 1
    result_df = slice.groupby('submitter').sum().sort_values(['execution_time'], ascending=False).reset_index()

    # Convert execution_time from timedelta to hours for proper display
    result_df['execution_time_hours'] = result_df['execution_time'].dt.total_seconds() / 3600

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Submitter', 'Execution Time (hours)', 'Number of Jobs']),
        cells=dict(values=[result_df.submitter, result_df['execution_time_hours'].round(2), result_df['number_of_jobs']]))
    ])
    fig.update_layout(title_text='Execution time and job count by submitter')

    html_file_path = os.path.join(output_path, "execution_time_by_submitter.html")
    fig.write_html(html_file_path)

def plot_utilization_per_device_type(jobs, output_path):
    """
    Generates a stacked horizontal bar chart showing device utilization by device type.

    Args:
        jobs (pandas.DataFrame): The DataFrame containing the job data.
        output_path (str): The path to save the HTML file.
    """
    # Convert time columns to datetime
    jobs['start_time'] = pd.to_datetime(jobs['start_time'])
    jobs['end_time'] = pd.to_datetime(jobs['end_time'])

    # Calculate total duration of the dataset in hours
    total_duration_dataset = (jobs['end_time'].max() - jobs['start_time'].min()).total_seconds() / 3600

    # Calculate the date range of the dataset
    start_date = jobs['start_time'].min().strftime('%Y-%m-%d %H:%M')
    end_date = jobs['end_time'].max().strftime('%Y-%m-%d %H:%M')

    # Group the data by 'requested_device_type'
    grouped_jobs = jobs.groupby('requested_device_type')

    # Initialize a list to store data for each 'requested_device_type'
    data_list = []

    # Iterate over each group of 'requested_device_type'
    for device_type, device_group in grouped_jobs:

        device_group['duration'] = (device_group['end_time'] - device_group['start_time']).dt.total_seconds() / 3600

        # Calculate the adapted total duration for this 'requested_device_type' by multiplying by the number of 'actual_device'
        total_duration_device = total_duration_dataset * device_group['actual_device'].nunique()

        # Calculate total durations for test jobs and health checks for this 'requested_device_type'
        health_checks_duration = device_group[device_group['submitter'] == 'lava-health']['duration'].sum()
        testjobs_duration = device_group[device_group['submitter'] != 'lava-health']['duration'].sum()

        # Calculate the inactive duration for this 'requested_device_type'
        inactive_duration = total_duration_device - health_checks_duration - testjobs_duration
        # Ensure that the inactive duration is not negative
        inactive_duration = max(inactive_duration, 0)

        # Calculate percentages for each category for this 'requested_device_type'
        health_checks_percent = (health_checks_duration / total_duration_device) * 100
        testjobs_percent = (testjobs_duration / total_duration_device) * 100
        inactive_percent = (inactive_duration / total_duration_device) * 100

        # Store the data in a dictionary
        device_data = {
            'requested_device_type': device_type,
            'total_duration_device': total_duration_device,
            'num_devices': device_group['actual_device'].nunique(),
            'health_checks_duration': health_checks_duration,
            'testjobs_duration': testjobs_duration,
            'inactive_duration': inactive_duration,
            'health_checks_percent': health_checks_percent,
            'testjobs_percent': testjobs_percent,
            'inactive_percent': inactive_percent
        }

        # Add this dictionary to the list
        data_list.append(device_data)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data_list)

    # Create the graph with Plotly
    fig = go.Figure()

    # Add bars for each category for each 'requested_device_type'
    for category in ['health_checks_percent', 'testjobs_percent', 'inactive_percent']:
        fig.add_trace(go.Bar(
            y=df['requested_device_type'],
            x=df[category],
            name=category.split('_')[0].capitalize(),  # Use the category name without '_percent'
            orientation='h',
            customdata=df[['num_devices', 'total_duration_device', category.replace('percent', 'duration')]],
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Percentage: %{x:.2f}%<br>' +
                'Number of Devices: %{customdata[0]}<br>' +
                'Total Duration Device: %{customdata[1]:.2f} hours<br>' +
                'Duration: %{customdata[2]:.2f} hours<extra></extra>'
            )
        ))

    # Format the graph
    fig.update_layout(
        barmode='stack',
        title=f'Distribution of Utilization by duration for Device Type (Percentage) ({start_date} to {end_date}, 100% = {total_duration_dataset:.2f} hours)',
        xaxis_title='Percentage of Time',
        yaxis_title='Device Type',
        legend_title='Category'
    )

    html_file_path = os.path.join(output_path, "utilization_per_device_type.html")
    fig.write_html(html_file_path)

def plot_weekly_utilization_device_type(jobs, output_path):
    """
    Generates a dotted line chart showing weekly device utilization by device type.

    Args:
        jobs (pandas.DataFrame): The DataFrame containing the job data.
        output_path (str): The path to save the HTML file.
    """
    # Convert time columns to datetime
    jobs['start_time'] = pd.to_datetime(jobs['start_time'])
    jobs['end_time'] = pd.to_datetime(jobs['end_time'])

    # Remove rows with null values in 'start_time' or 'end_time'
    jobs = jobs.dropna(subset=['start_time', 'end_time'])

    # Calculate the date range of the dataset
    start_date = jobs['start_time'].min()
    end_date = jobs['end_time'].max()

    # Remove timezone info if present
    if start_date.tzinfo is not None:
        start_date = start_date.tz_localize(None)
    if end_date.tzinfo is not None:
        end_date = end_date.tz_localize(None)

    # Group data by 'requested_device_type' and full week
    jobs['week'] = (jobs['start_time'] - pd.to_timedelta(jobs['start_time'].dt.weekday, unit='D')).dt.to_period('W').apply(lambda r: r.start_time).astype('datetime64[ns]')

    # Ensure 'week' is naive datetime
    if jobs['week'].dt.tz is not None:
        jobs['week'] = jobs['week'].dt.tz_localize(None)

    grouped_jobs = jobs.groupby(['requested_device_type', 'week'])

    # Initialize a list to store data for each 'requested_device_type'
    data_list = []

    # Number of hours in a week
    hours_in_week = 7 * 24

    # Iterate over each group of 'requested_device_type' and week
    for (device_type, week), device_group in grouped_jobs:
        device_group['duration'] = (device_group['end_time'] - device_group['start_time']).dt.total_seconds() / 3600

        # Calculate the total duration of jobs for this 'requested_device_type'
        jobs_duration = device_group['duration'].sum()

        # Calculate the number of unique devices for this type and week
        num_devices = device_group['actual_device'].nunique()

        # Calculate the total possible duration for this 'requested_device_type' and week
        total_duration_device = num_devices * hours_in_week

        # Calculate the percentage of jobs for this 'requested_device_type'
        if total_duration_device > 0:  # Avoid division by zero
            jobs_percent = (jobs_duration / total_duration_device) * 100
        else:
            jobs_percent = 0

        # Store data in a dictionary
        device_data = {
            'requested_device_type': device_type,
            'week': week,
            'jobs_percent': jobs_percent,
            'num_devices': num_devices,
            'total_duration_device': total_duration_device,
            'jobs_duration': jobs_duration
        }

        # Add this dictionary to the list
        data_list.append(device_data)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data_list)

    # Create the graph with Plotly
    fig = go.Figure()

    # Add dotted lines for the jobs
    for device_type in df['requested_device_type'].unique():
        device_data = df[df['requested_device_type'] == device_type]
        fig.add_trace(go.Scatter(
            x=device_data['week'],
            y=device_data['jobs_percent'],
            mode='lines+markers',
            name=f"{device_type}",
            customdata=device_data[['num_devices', 'total_duration_device', 'jobs_duration']],
            hovertemplate=(
                '<b>%{y:.2f}%</b><br>' +
                'Week: %{x}<br>' +
                'Number of devices: %{customdata[0]}<br>' +
                'Total device duration: %{customdata[1]:.2f} hours<br>' +
                'Duration: %{customdata[2]:.2f} hours<extra></extra>'
            ),
            line=dict(dash='dot')
        ))

    # Format the graph
    fig.update_layout(
        title=f'Weekly Usage by Requested Device Type ({start_date.strftime("%Y-%m-%d %H:%M")} to {end_date.strftime("%Y-%m-%d %H:%M")})',
        xaxis_title='Week',
        yaxis_title='Usage Percentage',
        legend_title='Category'
    )

    html_file_path = os.path.join(output_path, "Weekly_utilization_device_type.html")
    fig.write_html(html_file_path)


def plot_latency_by_device_type_over_time(jobs, output_path):
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    interval = pd.date_range(start=start_time, end=end_time, freq='8H')
    device_types = sorted(jobs.requested_device_type.unique())

    def accumulate_by_time_interval(date_range, device_types):
        step = date_range[1] - date_range[0]

        data = { 'date': [] }
        prev = {}
        for dt in device_types:
            data[dt] = []
            prev[dt] = 0.0

        for start in date_range:
            data['date'].append(start)
            #start -= step / 2
            end = start + step

            window = jobs[ (jobs.submit_time > start) & (jobs.submit_time < end)]
            for dt in device_types:
                subwindow = window[window.requested_device_type == dt]
                if len(subwindow):
                    mx = subwindow.latency.max()

                    if pd.isnull(mx):
                        mx = (step * 0)

                    # Graph in hours!
                    t = mx.total_seconds() / (60*60)
                    prev[dt] = t
                    data[dt].append(t)
                else:
                    # Nothing sumbitted in this interval so take the previous value
                    # and subtract the step interval from it (and use that if the value
                    # is positive)
                    t = prev[dt] - (step.total_seconds() / (24*60*60))
                    t = max(t, 0.0)
                    prev[dt] = t
                    data[dt].append(t)

        return pd.DataFrame(data)

    data = accumulate_by_time_interval(interval, device_types)

    greatest_percent = data.set_index(['date']).max().max()
    data_long = data.melt(id_vars='date', value_vars=device_types, var_name='Device Type', value_name='Latency')

    fig = px.line(data_long, x='date', y='Latency', color='Device Type',
                  title='Latency by Device Type Over Time',
                  labels={'date': 'Date', 'Latency': 'Latency (hours)'})

    fig.update_layout(xaxis={'type': 'category'}, height=600)  # Adjusting the height

    # Save the figure as an HTML file
    html_file_path = os.path.join(output_path, "Latency_by_device_type_over_Time.html")
    fig.write_html(html_file_path)

def plot_queue_length_over_time(events, output_path):
    # Make a copy to avoid modifying the original dataframe
    events = events.copy()

    # Convert timestamp to datetime
    events['timestamp'] = pd.to_datetime(events['timestamp'])

    start_time = events['timestamp'].min()
    end_time = events['timestamp'].max()

    # Use >= and <= to include all data points
    window = events[(events['timestamp'] >= start_time) & (events['timestamp'] <= end_time)]

    fig = px.line(window, x='timestamp', y='queue_length', title='Queue Length Over Time')
    fig.update_layout(yaxis_title='Queue Length')
    html_file_path = os.path.join(output_path, "Queue_Length_Over_Time.html")
    fig.write_html(html_file_path)

def plot_results_count_health_checks_actual_device(jobs, output_path):
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.start_time > start_time) & (jobs.start_time < end_time) & (pd.notnull(jobs.end_time)) & (jobs.submitter == 'lava-health')]
    # Convert the time columns to datetime
    window['start_time'] = pd.to_datetime(window['start_time'])
    window['end_time'] = pd.to_datetime(window['end_time'])

    # Calculate duration in hours
    window['duration'] = (window['end_time'] - window['start_time']).dt.total_seconds() / 3600

    # Group the data by requested_device_type
    groups = window[['requested_device_type', 'health', 'duration']].groupby('requested_device_type')

    # Create a figure with all data and buttons
    fig = go.Figure()

    # Lists to store the buttons
    buttons = []

    # Total number of traces
    total_traces = 0

    # Dictionary to keep track of trace visibility for each device type
    trace_visibility = {}

    # Color mapping for health statuses
    color_map = {
        'incomplete': 'red',
        'complete': 'green',
        'canceled': 'blue'
    }

    # Function to normalize health status strings
    def normalize_status(status):
        return status.strip().lower()

    for idx, (device_type, data_group) in enumerate(groups):
        
        # Filter the data for actual_devices with the current requested_device_type
        filtered_data = window[window['requested_device_type'] == device_type]

        # Group the data by actual_device
        actual_device_groups = filtered_data.groupby('actual_device')

        # Convert the keys of groups.groups into a list for subplot_titles
        subplot_titles = list(actual_device_groups.groups.keys())

        # Create a subplot with 6 rows and 2 columns, all of type 'pie'
        n_devices = len(subplot_titles)
        n_cols = 2
        n_rows = math.ceil(n_devices / n_cols)

        subplot_fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'pie'} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )   


        # Calculate the total duration for the requested_device_type
        total_duration_device_type = filtered_data['duration'].sum()
        total_health_checks_device_type = filtered_data.shape[0]

        # Loop through each actual_device and add a pie chart to the subplot
        for i, (device, device_group) in enumerate(actual_device_groups):
            counts = device_group['health'].value_counts()
            percentages = (counts / counts.sum() * 100).round(1)
            total_health_checks = counts.sum()
            
            # Calculate total durations for each health status
            total_durations = device_group.groupby('health')['duration'].sum()
            
            # Ensure the order of total_durations matches the order of counts and percentages
            total_durations = total_durations.reindex(counts.index)

            df = pd.DataFrame({'health': counts.index, 'count': counts.values, 'percentage': percentages.values, 'total_duration': total_durations.values})

            # Normalize health statuses
            df['health_normalized'] = df['health'].apply(normalize_status)

            # Get colors for the current health statuses
            colors = [color_map[normalize_status(status)] for status in df['health']]

            # Create a combined hover text with total duration in hours
            df['hover_text'] = df.apply(lambda row: f"Health: {row['health']}<br>Count: {row['count']}<br>Percentage: {row['percentage']}%<br>Total Duration: {row['total_duration']:.2f} hours", axis=1)

            # Create a pie chart for the current actual_device
            pie = go.Pie(
                labels=df['health'], 
                values=df['percentage'], 
                title=f'{device} (Total Health Checks: {total_health_checks}, Total Duration: {total_durations.sum():.2f} hours)', 
                name=device, 
                visible=(idx == 0), 
                marker=dict(colors=colors), 
                hovertext=df['hover_text'], 
                hoverinfo="label+percent+name+text"
            )
            
            # Calculate subplot position based on row and column
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Add pie chart to the subplot
            subplot_fig.add_trace(pie, row=row, col=col)
        
        # Add all traces from the subplot to the main figure
        for trace in subplot_fig.data:
            fig.add_trace(trace)
        
        # Calculate the number of traces for the current subplot
        num_traces = len(subplot_fig.data)

        # Initialize visibility list for the current device type
        visibility_list = [False] * window['actual_device'].nunique()
        
        # Set the visibility of the current traces to True
        for i in range(total_traces, total_traces + num_traces):
            visibility_list[i] = True
        
        # Keep track of trace visibility for the current device type
        trace_visibility[device_type] = visibility_list
        
        # Create a button for the requested_device_type
        button = dict(
            label=device_type,
            method='update',
            args=[
                {'visible': trace_visibility[device_type]},
                {"title": f'Results count of health checks for {device_type} (Total Health Checks: {total_health_checks_device_type}, Total Duration: {total_duration_device_type:.2f} hours)'}
            ]
        )
        buttons.append(button)
        
        # Update the total number of traces
        total_traces += num_traces

    # Add the buttons to the figure
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons
        )],
        height=2000,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    html_file_path = os.path.join(output_path, "Results_count_health_checks_actual_device.html")
    fig.write_html(html_file_path)

def plot_results_count_health_checks_device_type(jobs, output_path):
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.start_time > start_time) & (jobs.start_time < end_time) & (pd.notnull(jobs.end_time)) & (jobs.submitter == 'lava-health')]
    # Convert the time columns to datetime
    window['start_time'] = pd.to_datetime(window['start_time'])
    window['end_time'] = pd.to_datetime(window['end_time'])

    # Calculate duration in hours
    window['duration'] = (window['end_time'] - window['start_time']).dt.total_seconds() / 3600

    # Group the data by requested_device_type and health
    groups = window[['requested_device_type', 'health', 'duration']].groupby('requested_device_type')

    # Lists to store the traces of each pie chart and the buttons
    data = []
    buttons = []
    first_device_type = True

    # Color mapping for health statuses
    color_map = {
        'incomplete': 'red',
        'complete': 'green',
        'canceled': 'blue'
    }

    # Create a trace for each requested_device_type and add to traces and buttons
    for i, (device_type, data_group) in enumerate(groups):
        counts = data_group['health'].value_counts()
        percentages = (counts / counts.sum() * 100).round(1)
        total_health_checks = counts.sum()
        
        # Calculate total durations for each health status
        total_durations = data_group.groupby('health')['duration'].sum()
        # Ensure the order of total_durations matches the order of counts and percentages
        total_durations = total_durations.reindex(counts.index)
        
        # Calculate the total duration for the device type
        total_duration_device_type = data_group['duration'].sum()

        df = pd.DataFrame({'health': counts.index, 'count': counts.values, 'percentage': percentages.values, 'total_duration': total_durations.values})

        # Normalize health statuses for consistent color mapping
        df['health_normalized'] = df['health'].str.strip().str.lower()

        # Create a combined hover text with total duration in hours
        df['hover_text'] = df.apply(lambda row: f"Health: {row['health']}<br>Count: {row['count']}<br>Percentage: {row['percentage']}%<br>Total Duration: {row['total_duration']:.2f} hours", axis=1)
        # Create the pie chart using plotly.express.pie and apply color mapping
        fig = px.pie(
            df, 
            names='health', 
            values='percentage', 
            title=f'Distribution of health for {device_type} (Total Health Checks: {total_health_checks}, Total Duration: {total_duration_device_type:.2f} hours)',
            color='health_normalized',
            color_discrete_map=color_map,
            hover_name='hover_text'  # Use the combined hover text
        )

        # Add each pie chart trace to the data list
        for trace in fig.data:
            trace.visible = first_device_type  # The first chart is visible initially, others are hidden
            data.append(trace)
        
        # Create a button for this device_type
        visibility = [False] * window['requested_device_type'].nunique()
        visibility[i] = True
        buttons.append(dict(
            label=device_type,
            method="update",
            args=[{"visible": visibility},
                  {"title": f'Results count of health checks for {device_type} (Total Health Checks: {total_health_checks}, Total Duration: {total_duration_device_type:.2f} hours)'}],
        ))
        
        first_device_type = False

    # Create the initial figure
    fig = go.Figure(data=data)

    # Add the buttons to the figure
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons
        )]
    )

    html_file_path = os.path.join(output_path, "Results_count_health_checks_device_type.html")
    fig.write_html(html_file_path)

def plot_results_count_jobs_actual_device(jobs, output_path):
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.start_time > start_time) & (jobs.start_time < end_time) & (pd.notnull(jobs.end_time))]
    # Convert the time columns to datetime
    window['start_time'] = pd.to_datetime(window['start_time'])
    window['end_time'] = pd.to_datetime(window['end_time'])

    # Calculate duration in hours
    window['duration'] = (window['end_time'] - window['start_time']).dt.total_seconds() / 3600

    # Group the data by requested_device_type
    groups = window[['requested_device_type', 'health', 'duration']].groupby('requested_device_type')

    # Create a figure with all data and buttons
    fig = go.Figure()

    # Lists to store the buttons
    buttons = []

    # Total number of traces
    total_traces = 0

    # Dictionary to keep track of trace visibility for each device type
    trace_visibility = {}

    # Color mapping for health statuses
    color_map = {
        'incomplete': 'red',
        'complete': 'green',
        'canceled': 'blue'
    }

    # Function to normalize health status strings
    def normalize_status(status):
        return status.strip().lower()

    for idx, (device_type, data_group) in enumerate(groups):
        
        # Filter the data for actual_devices with the current requested_device_type
        filtered_data = window[window['requested_device_type'] == device_type]

        # Group the data by actual_device
        actual_device_groups = filtered_data.groupby('actual_device')

        # Convert the keys of groups.groups into a list for subplot_titles
        subplot_titles = list(actual_device_groups.groups.keys())

        # Create a subplot with 6 rows and 2 columns, all of type 'pie'
        subplot_fig = make_subplots(
            rows=6, 
            cols=2, 
            subplot_titles=subplot_titles, 
            specs=[[{'type': 'pie'}, {'type': 'pie'}]] * 6,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        import math

        n_devices = len(subplot_titles)
        n_cols = 2
        n_rows = math.ceil(n_devices / n_cols)

        subplot_fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'pie'} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )


        # Calculate the total duration for the requested_device_type
        total_duration_device_type = filtered_data['duration'].sum()
        total_health_checks_device_type = filtered_data.shape[0]

        # Loop through each actual_device and add a pie chart to the subplot
        for i, (device, device_group) in enumerate(actual_device_groups):
            counts = device_group['health'].value_counts()
            
            # Calculate total durations for each health status
            total_durations = device_group.groupby('health')['duration'].sum()
            # Ensure the order of total_durations matches the order of counts
            total_durations = total_durations.reindex(counts.index)
            total_health_checks = counts.sum()
            
            df = pd.DataFrame({'health': counts.index, 'count': counts.values, 'total_duration': total_durations.values})

            # Normalize health statuses
            df['health_normalized'] = df['health'].apply(normalize_status)

            # Get colors for the current health statuses
            colors = [color_map[normalize_status(status)] for status in df['health']]

            # Create a combined hover text with count of jobs
            df['hover_text'] = df.apply(lambda row: f"Health: {row['health']}<br>Count: {row['count']}<br>Total Duration: {row['total_duration']:.2f} hours", axis=1)

            # Create a pie chart for the current actual_device
            pie = go.Pie(
                labels=df['health'], 
                values=df['total_duration'],  # Use total_duration for values
                title=f'{device} (Total jobs: {total_health_checks}, Total Duration: {total_durations.sum():.2f} hours)', 
                name=device, 
                visible=(idx == 0), 
                marker=dict(colors=colors), 
                hovertext=df['hover_text'], 
                hoverinfo="label+percent+name+text"
            )
            
            # Calculate subplot position based on row and column
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Add pie chart to the subplot
            subplot_fig.add_trace(pie, row=row, col=col)
        
        # Add all traces from the subplot to the main figure
        for trace in subplot_fig.data:
            fig.add_trace(trace)
        
        # Calculate the number of traces for the current subplot
        num_traces = len(subplot_fig.data)

        # Initialize visibility list for the current device type
        visibility_list = [False] * window['actual_device'].nunique()
        
        # Set the visibility of the current traces to True
        for i in range(total_traces, total_traces + num_traces):
            visibility_list[i] = True
        
        # Keep track of trace visibility for the current device type
        trace_visibility[device_type] = visibility_list
        
        # Create a button for the requested_device_type
        button = dict(
            label=device_type,
            method='update',
            args=[
                {'visible': trace_visibility[device_type]},
                {"title": f'Results count jobs for {device_type} (Total jobs: {total_health_checks_device_type}, Total Duration: {total_duration_device_type:.2f} hours)'}
            ]
        )
        buttons.append(button)
        
        # Update the total number of traces
        total_traces += num_traces

    # Add the buttons to the figure
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons
        )],
        height=2000,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    html_file_path = os.path.join(output_path, "Results_count_jobs_actual_device.html")
    fig.write_html(html_file_path)

def plot_results_count_jobs_device_type(jobs, output_path):
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.start_time > start_time) & (jobs.start_time < end_time) & (pd.notnull(jobs.end_time))]
    # Convert the time columns to datetime
    window['start_time'] = pd.to_datetime(window['start_time'])
    window['end_time'] = pd.to_datetime(window['end_time'])

    # Calculate duration in hours
    window['duration'] = (window['end_time'] - window['start_time']).dt.total_seconds() / 3600

    # Group the data by requested_device_type and health
    groups = window[['requested_device_type', 'health', 'duration']].groupby('requested_device_type')

    # Lists to store the traces of each pie chart and the buttons
    data = []
    buttons = []
    first_device_type = True

    # Color mapping for health statuses
    color_map = {
        'incomplete': 'red',
        'complete': 'green',
        'canceled': 'blue'
    }

    # Create a trace for each requested_device_type and add to traces and buttons
    for i, (device_type, data_group) in enumerate(groups):
        counts = data_group['health'].value_counts()
        percentages = (counts / counts.sum() * 100).round(1)
        total_jobs = counts.sum()
        
        # Calculate total durations for each health status
        total_durations = data_group.groupby('health')['duration'].sum()
        # Ensure the order of total_durations matches the order of counts and percentages
        total_durations = total_durations.reindex(counts.index)
        
        # Calculate the total duration for the device type
        total_duration_device_type = data_group['duration'].sum()

        df = pd.DataFrame({'health': counts.index, 'count': counts.values, 'percentage': percentages.values, 'total_duration': total_durations.values})

        # Normalize health statuses for consistent color mapping
        df['health_normalized'] = df['health'].str.strip().str.lower()

        # Create a combined hover text with total duration in hours
        df['hover_text'] = df.apply(lambda row: f"Health: {row['health']}<br>Count: {row['count']}<br>Percentage: {row['percentage']}%<br>Total Duration: {row['total_duration']:.2f} hours", axis=1)

        # Create the pie chart using plotly.express.pie and apply color mapping
        fig = px.pie(
            df, 
            names='health', 
            values='percentage', 
            title=f'Results of jobs for {device_type} (Total jobs: {total_jobs}, Total Duration: {total_duration_device_type:.2f} hours)',
            color='health_normalized',
            color_discrete_map=color_map,
            hover_name='hover_text'  # Use the combined hover text
        )

        # Add each pie chart trace to the data list
        for trace in fig.data:
            trace.visible = first_device_type  # The first chart is visible initially, others are hidden
            data.append(trace)
        
        # Create a button for this device_type
        visibility = [False] * len(groups)
        visibility[i] = True
        buttons.append(dict(
            label=device_type,
            method="update",
            args=[{"visible": visibility},
                  {"title": f'Results count jobs for {device_type} (Total jobs: {total_jobs}, Total Duration: {total_duration_device_type:.2f} hours)'}],
        ))
        
        first_device_type = False

    # Create the initial figure
    fig = go.Figure(data=data)

    # Add the buttons to the figure
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons
        )]
    )

    html_file_path = os.path.join(output_path, "Results_count_jobs_device_type.html")
    fig.write_html(html_file_path)

def plot_duration_jobs_device_type(jobs, output_path):
    """
    Generates pie charts showing job duration by health status for each device type.

    Args:
        jobs (pandas.DataFrame): The DataFrame containing the job data.
        output_path (str): The path to save the HTML file.
    """
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.start_time > start_time) & (jobs.start_time < end_time) & (pd.notnull(jobs.end_time))]

    # Convert the time columns to datetime
    window['start_time'] = pd.to_datetime(window['start_time'])
    window['end_time'] = pd.to_datetime(window['end_time'])

    # Calculate duration in hours
    window['duration'] = (window['end_time'] - window['start_time']).dt.total_seconds() / 3600

    # Group the data by requested_device_type and health
    groups = window[['requested_device_type', 'health', 'duration']].groupby('requested_device_type')

    # Lists to store the traces of each pie chart and the buttons
    data = []
    buttons = []
    first_device_type = True

    # Color mapping for health statuses
    color_map = {
        'incomplete': 'red',
        'complete': 'green',
        'canceled': 'blue'
    }

    # Create a trace for each requested_device_type and add to traces and buttons
    for i, (device_type, data_group) in enumerate(groups):
        counts = data_group['health'].value_counts()
        # Calculate total durations for each health status
        total_durations = data_group.groupby('health')['duration'].sum()
        total_jobs = counts.sum()
        total_duration_device_type = total_durations.sum()

        # Ensure the order of total_durations matches the order of counts
        total_durations = total_durations.reindex(counts.index)

        df = pd.DataFrame({'health': counts.index, 'count': counts.values, 'total_duration': total_durations.values})

        # Normalize health statuses for consistent color mapping
        df['health_normalized'] = df['health'].str.strip().str.lower()

        # Create a combined hover text with count of jobs
        df['hover_text'] = df.apply(lambda row: f"Health: {row['health']}<br>Count: {row['count']}<br>Total Duration: {row['total_duration']:.2f} hours", axis=1)

        # Create the pie chart using plotly.express.pie and apply color mapping
        fig = px.pie(
            df,
            names='health',
            values='total_duration',  # Use total_duration for values
            title=f'Duration of jobs for {device_type} (Total jobs: {total_jobs}, Total Duration: {total_duration_device_type:.2f} hours)',
            color='health_normalized',
            color_discrete_map=color_map,
            hover_name='hover_text'  # Use the combined hover text
        )

        # Add each pie chart trace to the data list
        for trace in fig.data:
            trace.visible = first_device_type  # The first chart is visible initially, others are hidden
            data.append(trace)

        # Create a button for this device_type
        visibility = [False] * len(groups)
        visibility[i] = True
        buttons.append(dict(
            label=device_type,
            method="update",
            args=[{"visible": visibility},
                  {"title": f'Duration of jobs for {device_type} (Total jobs: {total_jobs}, Total Duration: {total_duration_device_type:.2f} hours)'}],
        ))

        first_device_type = False

    # Create the initial figure
    fig = go.Figure(data=data)

    # Add the buttons to the figure
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons
        )]
    )

    html_file_path = os.path.join(output_path, "duration_jobs_device_type.html")
    fig.write_html(html_file_path)

def plot_duration_jobs_actual_device(jobs, output_path):
    """
    Generates a grid of pie charts showing job duration by health status for each actual device.

    Args:
        jobs (pandas.DataFrame): The DataFrame containing the job data.
        output_path (str): The path to save the HTML file.
    """
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.start_time > start_time) & (jobs.start_time < end_time) & (pd.notnull(jobs.end_time))]

    # Convert the time columns to datetime
    window['start_time'] = pd.to_datetime(window['start_time'])
    window['end_time'] = pd.to_datetime(window['end_time'])

    # Calculate duration in hours
    window['duration'] = (window['end_time'] - window['start_time']).dt.total_seconds() / 3600

    # Group the data by requested_device_type
    groups = window[['requested_device_type', 'health', 'duration']].groupby('requested_device_type')

    # Create a figure with all data and buttons
    fig = go.Figure()

    # Lists to store the buttons
    buttons = []

    # Total number of traces
    total_traces = 0

    # Dictionary to keep track of trace visibility for each device type
    trace_visibility = {}

    # Color mapping for health statuses
    color_map = {
        'incomplete': 'red',
        'complete': 'green',
        'canceled': 'blue'
    }

    # Function to normalize health status strings
    def normalize_status(status):
        return status.strip().lower()

    for idx, (device_type, data_group) in enumerate(groups):

        # Filter the data for actual_devices with the current requested_device_type
        filtered_data = window[window['requested_device_type'] == device_type]

        # Group the data by actual_device
        actual_device_groups = filtered_data.groupby('actual_device')

        # Convert the keys of groups.groups into a list for subplot_titles
        subplot_titles = list(actual_device_groups.groups.keys())

        # Create a subplot with dynamic rows and 2 columns, all of type 'pie'
        n_devices = len(subplot_titles)
        n_cols = 2
        n_rows = math.ceil(n_devices / n_cols)

        subplot_fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'pie'} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # Calculate the total duration for the requested_device_type
        total_duration_device_type = filtered_data['duration'].sum()
        total_jobs_device_type = filtered_data.shape[0]

        # Loop through each actual_device and add a pie chart to the subplot
        for i, (device, device_group) in enumerate(actual_device_groups):
            counts = device_group['health'].value_counts()

            # Calculate total durations for each health status
            total_durations = device_group.groupby('health')['duration'].sum()
            # Ensure the order of total_durations matches the order of counts
            total_durations = total_durations.reindex(counts.index)
            total_jobs = counts.sum()

            df = pd.DataFrame({'health': counts.index, 'count': counts.values, 'total_duration': total_durations.values})

            # Normalize health statuses
            df['health_normalized'] = df['health'].apply(normalize_status)

            # Get colors for the current health statuses
            colors = [color_map[normalize_status(status)] for status in df['health']]

            # Create a combined hover text with count of jobs
            df['hover_text'] = df.apply(lambda row: f"Health: {row['health']}<br>Count: {row['count']}<br>Total Duration: {row['total_duration']:.2f} hours", axis=1)

            # Create a pie chart for the current actual_device
            pie = go.Pie(
                labels=df['health'],
                values=df['total_duration'],  # Use total_duration for values
                title=f'{device} (Total jobs: {total_jobs}, Total Duration: {total_durations.sum():.2f} hours)',
                name=device,
                visible=(idx == 0),
                marker=dict(colors=colors),
                hovertext=df['hover_text'],
                hoverinfo="label+percent+name+text"
            )

            # Calculate subplot position based on row and column
            row = i // 2 + 1
            col = i % 2 + 1

            # Add pie chart to the subplot
            subplot_fig.add_trace(pie, row=row, col=col)

        # Add all traces from the subplot to the main figure
        for trace in subplot_fig.data:
            fig.add_trace(trace)

        # Calculate the number of traces for the current subplot
        num_traces = len(subplot_fig.data)

        # Initialize visibility list for the current device type
        visibility_list = [False] * window['actual_device'].nunique()

        # Set the visibility of the current traces to True
        for i in range(total_traces, total_traces + num_traces):
            visibility_list[i] = True

        # Keep track of trace visibility for the current device type
        trace_visibility[device_type] = visibility_list

        # Create a button for the requested_device_type
        button = dict(
            label=device_type,
            method='update',
            args=[
                {'visible': trace_visibility[device_type]},
                {"title": f'Duration of jobs for {device_type} (Total jobs: {total_jobs_device_type}, Total Duration: {total_duration_device_type:.2f} hours)'}
            ]
        )
        buttons.append(button)

        # Update the total number of traces
        total_traces += num_traces

    # Add the buttons to the figure
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons
        )],
        height=2000,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    html_file_path = os.path.join(output_path, "duration_jobs_actual_device.html")
    fig.write_html(html_file_path)

def plot_scheduling_latency_over_time(jobs, output_path):
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.requested_device_type == 'dragonboard-410c') & 
                  (jobs.start_time > start_time) & 
                  (jobs.start_time < end_time)]

    fig = px.line(window, x='submit_time', y='latency_in_minutes', title='Scheduling Latency Over Time')
    fig.update_layout(yaxis_title='Scheduling Latency (in minutes)')
    html_file_path = os.path.join(output_path, "Scheduling_Latency_Over_Time.html")
    fig.write_html(html_file_path)

def _create_timeline(jobs, output_path, color_by, title, filename, color_discrete_map=None):
    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.start_time > start_time) & (jobs.start_time < end_time) & (pd.notnull(jobs.end_time))].copy()

    if window[color_by].dtype == 'object':
        window[color_by] = window[color_by].str.slice(0, 50)

    fig = px.timeline(window, x_start="start_time", x_end="end_time", y="requested_device_type", color=color_by,
                    custom_data=['id', 'submitter', 'description'],
                    color_discrete_map=color_discrete_map)

    fig.update_traces(hovertemplate=
                    "Job ID: %{customdata[0]}<br>" +
                    "Submitter: %{customdata[1]}<br>" +
                    "Description: %{customdata[2]}"
                    )

    fig.update_layout(title_text=title)
    html_file_path = os.path.join(output_path, filename)
    fig.write_html(html_file_path)

def plot_timeline_by_id(jobs, output_path):
    _create_timeline(jobs, output_path, "id", "Timeline by ID", "Timeline_by_id.html", color_discrete_map=None)

def plot_timeline_by_job_result(jobs, output_path):
    _create_timeline(jobs, output_path, "health", "Timeline by Job Result", "Timeline_by_job_result.html", color_discrete_map=None)

def plot_timeline_by_submitter(jobs, output_path):
    _create_timeline(jobs, output_path, "submitter", "Timeline by Submitter", "Timeline_by_submitter.html", color_discrete_map=None)

def plot_timeline_by_priority(jobs, output_path):
    def priority_group(priority):
        if priority > 70:
            return 'High'
        elif 30 <= priority <= 70:
            return 'Medium'
        else:
            return 'Low'

    jobs['priority_group'] = jobs['priority'].apply(priority_group)
    color_map = {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}
    _create_timeline(jobs, output_path, "priority_group", "Timeline by Priority", "Timeline_by_priority.html", color_discrete_map=color_map)

def plot_timeline_by_test_case_description(jobs, output_path):
    _create_timeline(jobs, output_path, "description", "Timeline by Test Case Description", "Timeline_by_Test_Case_Description.html", color_discrete_map=None)

def plot_timeline_by_test_suite_definition(jobs, output_path):
    _create_timeline(jobs, output_path, "definition", "Timeline by Test Suite Definition", "Timeline_by_Test_Suite_Definition.html", color_discrete_map=None)

def plot_timeline_by_execution_time(jobs, output_path):
    # Convert time columns to datetime
    jobs['start_time'] = pd.to_datetime(jobs['start_time'])
    jobs['end_time'] = pd.to_datetime(jobs['end_time'])

    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.end_time > start_time) & (jobs.start_time < end_time)].copy()

    # Add an execution time column in minutes
    window['execution_time_minutes'] = window['execution_time'].dt.total_seconds() / 60

    def assign_color(execution_time_minutes):
        if execution_time_minutes <= 300:
            return 'less than 300 minutes'
        elif execution_time_minutes <= 600:
            return 'between 300 and 600 minutes'
        else:
            return 'more than 600 minutes'

    window['Execution_periods'] = window['execution_time_minutes'].apply(assign_color)

    fig = px.timeline(window,
                      x_start="start_time", x_end="end_time", y="actual_device", color="Execution_periods", height=2000, title="Timeline by execution time",
                      hover_data={'id': True, 'execution_time_minutes': True, 'description': True},
                      color_discrete_map={'less than 300 minutes': 'green', 'between 300 and 600 minutes': 'orange', 'more than 600 minutes': 'red'})

    fig.update_yaxes(categoryorder="category ascending")
    html_file_path = os.path.join(output_path, "Timeline_by_Execution_time.html")
    fig.write_html(html_file_path)

def plot_timeline_by_latency(jobs, output_path):
    # Convert time columns to datetime
    jobs['start_time'] = pd.to_datetime(jobs['start_time'])
    jobs['end_time'] = pd.to_datetime(jobs['end_time'])

    start_time = jobs[['submit_time', 'start_time', 'end_time']].min().min()
    end_time = jobs[['submit_time', 'start_time', 'end_time']].max().max()
    window = jobs[(jobs.end_time > start_time) & (jobs.start_time < end_time)].copy()

    # Add a latency column in hours
    window['latency_hours'] = (window['latency'].dt.total_seconds() / 60) / 60

    def assign_color(latency_hours):
        if latency_hours <= 20:
            return 'less than 20 hours'
        elif latency_hours <= 40:
            return 'between 20 and 40 hours'
        else:
            return 'more than 40 hours'

    window['Latency_periods'] = window['latency_hours'].apply(assign_color)

    fig = px.timeline(window,
                      x_start="start_time", x_end="end_time", y="actual_device", color="Latency_periods", height=2000, title="Timeline by latency",
                      hover_data={'id': True, 'latency_hours': True, 'description': True},
                      color_discrete_map={'less than 20 hours': 'green', 'between 20 and 40 hours': 'orange', 'more than 40 hours': 'red'})

    fig.update_yaxes(categoryorder="category ascending")
    html_file_path = os.path.join(output_path, "Timeline_by_Latency.html")
    fig.write_html(html_file_path)

def generate_index_html(output_dir):
    """
    Generates an index.html file that displays all the generated plots.

    Args:
        output_dir (str): The path to the directory containing the HTML files.
    """
    html_files = [f for f in os.listdir(output_dir) if f.endswith('.html') and f != 'index.html']
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write('<html>\n')
        f.write('<head><title>LAVA Job Visualizer</title></head>\n')
        f.write('<body>\n')
        f.write('<h1>LAVA Job Visualizer</h1>\n')
        
        for html_file in html_files:
            f.write(f'<h2>{html_file}</h2>\n')
            f.write(f'<iframe src="{html_file}" width="100%" height="600px"></iframe>\n')
            f.write('<hr>\n')
            
        f.write('</body>\n')
        f.write('</html>\n')
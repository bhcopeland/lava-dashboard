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

    fig = go.Figure(data=[go.Table(
        header=dict(values=['Submitter', 'Execution Time', 'Number of Jobs']),
        cells=dict(values=[result_df.submitter, result_df['execution_time'], result_df['number_of_jobs']]))
    ])
    fig.update_layout(title_text='Execution time and job count by submitter')

    html_file_path = os.path.join(output_path, "execution_time_by_submitter.html")
    fig.write_html(html_file_path)

def plot_cost_per_device_type(jobs, output_path):
    # Convert time columns to datetime if not already done
    jobs['start_time'] = pd.to_datetime(jobs['start_time'])
    jobs['end_time'] = pd.to_datetime(jobs['end_time'])

    # Calculate total duration of the dataset in hours
    total_duration_dataset = (jobs['end_time'].max() - jobs['start_time'].min()).total_seconds() / 3600

    # Group by 'requested_device_type'
    grouped_jobs = jobs.groupby('requested_device_type')

    # Initialize a list to store data
    data_list = []

    # Iterate over each group of 'requested_device_type'
    for device_type, device_group in grouped_jobs:
        
        # Separate 'health checks' and 'test jobs'
        health_checks = device_group[device_group['submitter'] == 'lava-health']
        testjobs = device_group[device_group['submitter'] != 'lava-health']
        
        # Calculate total duration of 'health checks' and 'test jobs' for this 'requested_device_type'
        health_checks_duration = (health_checks['end_time'] - health_checks['start_time']).sum().total_seconds() / 3600
        testjobs_duration = (testjobs['end_time'] - testjobs['start_time']).sum().total_seconds() / 3600
        
        # Store data in a dictionary
        device_data = {
            'requested_device_type': device_type,
            'health_checks_duration': health_checks_duration,
            'testjobs_duration': testjobs_duration,
        }
        
        # Append this dictionary to the list
        data_list.append(device_data)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data_list)

    # Calculate cost in dollars for each device type
    df['cost'] = df['testjobs_duration'] * 0.57 + df['health_checks_duration'] * 0.57

    # Create the graph with Plotly
    fig = go.Figure()

    # Add a bar for the cost for each device type
    fig.add_trace(go.Bar(
        x=df['requested_device_type'],
        y=df['cost'],
        text=df['cost'].round(2),  # Text displayed on hover
        hoverinfo='text+y',  # Additional information on hover
        marker_color='purple',  # Bar color
    ))

    # Format the graph
    fig.update_layout(
        title='Cost in Dollars by Device Type',
        xaxis_title='Device Type',
        yaxis_title='Cost (USD)',
    )

    html_file_path = os.path.join(output_path, "Cost_per_devict_type.html")
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
    start_time = events[['timestamp']].min().min()
    end_time = events[['timestamp']].max().max()
    window = events[(events['timestamp'] > start_time) & (events['timestamp'] < end_time)]

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
    _create_timeline(jobs, output_path, "execution_time", "Timeline by Execution Time", "Timeline_by_Execution_time.html", color_discrete_map=None)

def plot_timeline_by_latency(jobs, output_path):
    _create_timeline(jobs, output_path, "latency", "Timeline by Latency", "Timeline_by_Latency.html", color_discrete_map=None)

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
# LAVA Job Visualizer

This project provides a set of Python scripts to analyze and visualize LAVA job data.

## Data Collection

To collect data from your LAVA server, you need to run the exporter job that generates the JSON data file required by this visualizer.

The exporter job is available in the LAVA server codebase:
- **Merge Request**: https://gitlab.com/lava/lava/-/merge_requests/2992

This exporter job will:
- Extract job data including submission times, execution times, device information, and health status
- Generate event data for queue length tracking over time
- Output the data in JSON format suitable for this visualizer

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the LAVA exporter job on your LAVA server to generate the data file (see **Data Collection** section above).
2. Place your LAVA job data file (in JSON format) in the `data` directory.
3. Run the main script:
   ```
   python src/main.py data/<your-data-file.json>
   ```

The generated graphs will be saved in the `graphs` directory.

## Generated Visualizations

The visualizer creates the following graphs:

- **Breakdown of Submitters**: Pie chart showing job distribution by submitter
- **Execution Time by Submitter**: Table showing execution time and job count per submitter
- **Utilization per Device Type**: Stacked bar chart showing device utilization percentages
- **Weekly Utilization**: Line chart showing weekly device utilization trends
- **Queue Length Over Time**: Line chart tracking queue length changes
- **Latency Metrics**: Various visualizations for scheduling latency
- **Timeline Views**: Gantt-style charts showing job timelines by different criteria (ID, submitter, priority, latency, execution time)
- **Health Check Results**: Pie charts showing health check outcomes by device
- **Job Duration**: Pie charts showing job duration by health status

All graphs are interactive HTML files that can be opened in a web browser.

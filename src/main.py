
import argparse
import os
from data_loader import load_and_process_data
from plotting import (
    plot_submitter_breakdown,

    plot_execution_time_by_submitter,
    plot_utilization_per_device_type,
    plot_weekly_utilization_device_type,

    plot_latency_by_device_type_over_time,
    plot_queue_length_over_time,
    plot_results_count_health_checks_actual_device,
    plot_results_count_health_checks_device_type,
    plot_results_count_jobs_actual_device,
    plot_results_count_jobs_device_type,
    plot_duration_jobs_device_type,
    plot_duration_jobs_actual_device,
    plot_scheduling_latency_over_time,
    plot_timeline_by_id,
    plot_timeline_by_job_result,
    plot_timeline_by_submitter,
    plot_timeline_by_priority,
    plot_timeline_by_test_case_description,
    plot_timeline_by_test_suite_definition,
    plot_timeline_by_execution_time,
    plot_timeline_by_latency,
    generate_index_html
)

def main():
    """Main function to run the LAVA job visualizer."""
    parser = argparse.ArgumentParser(description='LAVA Job Visualizer')
    parser.add_argument('data_file', type=str, help='Path to the LAVA job data file (JSON format)')
    args = parser.parse_args()

    # Check if the data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at: {args.data_file}")
        print("Please make sure the data file exists and you have provided the correct path.")
        print("It is recommended to place your data file in the 'data' directory.")
        return

    # Create the output directory if it doesn't exist
    output_dir = 'graphs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and process the data
    print("Loading and processing data...")
    jobs, events = load_and_process_data(args.data_file)
    print("Data loaded and processed successfully.")

    # Generate the plots
    print("Generating plots...")
    plot_submitter_breakdown(jobs, output_dir)

    plot_execution_time_by_submitter(jobs, output_dir)
    plot_utilization_per_device_type(jobs, output_dir)
    plot_weekly_utilization_device_type(jobs, output_dir)

    plot_latency_by_device_type_over_time(jobs, output_dir)
    plot_queue_length_over_time(events, output_dir)
    plot_results_count_health_checks_actual_device(jobs, output_dir)
    plot_results_count_health_checks_device_type(jobs, output_dir)
    plot_results_count_jobs_actual_device(jobs, output_dir)
    plot_results_count_jobs_device_type(jobs, output_dir)
    plot_duration_jobs_device_type(jobs, output_dir)
    plot_duration_jobs_actual_device(jobs, output_dir)
    plot_scheduling_latency_over_time(jobs, output_dir)
    plot_timeline_by_id(jobs, output_dir)
    plot_timeline_by_job_result(jobs, output_dir)
    plot_timeline_by_submitter(jobs, output_dir)
    plot_timeline_by_priority(jobs, output_dir)
    plot_timeline_by_test_case_description(jobs, output_dir)
    plot_timeline_by_test_suite_definition(jobs, output_dir)
    plot_timeline_by_execution_time(jobs, output_dir)
    plot_timeline_by_latency(jobs, output_dir)
    print(f"Plots generated and saved in the '{output_dir}' directory.")

    # Generate the index.html file
    print("Generating index.html...")
    generate_index_html(output_dir)
    print("index.html generated successfully.")

if __name__ == '__main__':
    main()

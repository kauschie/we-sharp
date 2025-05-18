from tbparse import SummaryReader
from tensorboardX import SummaryWriter
import os

def filter_tbparse_events(input_dir, output_dir, min_step=None, max_step=None):
    """
    Filters TensorBoard logs using tbparse for reading and tensorboardX for writing.
    Args:
        input_dir (str): Directory containing original event files.
        output_dir (str): Directory to save filtered event files.
        min_step (int): Minimum step to filter out (inclusive).
        max_step (int): Maximum step to filter out (inclusive).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read the TensorBoard logs using tbparse
    reader = SummaryReader(input_dir)
    df = reader.scalars

    # Filter out rows with steps in the specified range
    filtered_df = df[~((df["step"] >= min_step) & (df["step"] <= max_step))]

    # Create a writer for saving the filtered data
    writer = SummaryWriter(logdir=output_dir)

    # Write filtered data back to event files
    for _, row in filtered_df.iterrows():
        writer.add_scalar(row["tag"], row["value"], row["step"])

    writer.close()
    print(f"Filtered event files saved to {output_dir}")

# Example Usage
filter_tbparse_events(
    input_dir="../logs/coarse",
    output_dir="../logs/filtered_coarse",
    min_step=7001,
    max_step=8000
)

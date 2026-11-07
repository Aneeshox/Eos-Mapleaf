import re
import csv

def simple_extract(input_file, output_file):
    """Simple function to extract first two numbers from each line."""
    
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        
        for line in f_in:
            # Find all numbers in the line
            numbers = re.findall(r'-?\d+\.?\d*', line)
            
            # If we have at least one number
            if numbers:
                # Take first number, and second if it exists
                row = [numbers[0]]
                if len(numbers) >= 2:
                    row.append(numbers[1])
                else:
                    row.append('')
                writer.writerow(row)
# simple_extract('/Users/sabhu/Desktop/MAPLEAF/Tests/3P5k4P15b1P8ox.txt','/opt/anaconda3/envs/ROCKET/lib/python3.11/rocketpy/3P5k4P15b1P8ox.csv')
# Example usage:
simple_extract('rocketpy/eos/thrust.csv', 'rocketpy/eos/thrust1.csv')
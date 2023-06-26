input_file = "/home/antoine/KGene2Pheno/data/raw/local_celegans.txt"  # Replace with the path to your input file
output_file = "/home/antoine/KGene2Pheno/data/raw/local_celegans-test.txt"  # Replace with the desired output file path

# Read the input file
with open(input_file, "r") as file:
    lines = file.readlines()

# Process each line and convert the format
output_lines = []
for i, line in enumerate(lines):
    # Remove leading/trailing whitespace and split the line by spaces
    items = line.strip().split(" ")
    
    # Join the items with commas and prepend a comma to the line
    output_line = f"{i}," + ",".join(items) + "\n"
    output_lines.append(output_line)

# Write the output to a new file
with open(output_file, "w") as file:
    file.writelines(output_lines)
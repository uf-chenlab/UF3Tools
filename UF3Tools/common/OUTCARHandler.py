class OutcarCombiner:
    def __init__(self, original_outcar, restarted_outcar):
        self.original_outcar = original_outcar
        self.restarted_outcar = restarted_outcar
        self.combined_outcar = None

    def find_overlap_line(self):
        with open(self.restarted_outcar, 'r') as file:
            lines = file.readlines()
        
        for i, line in enumerate(lines):
            if "Iteration" in line:
                return i
        
        raise ValueError("No overlap found or file does not contain expected SCF iteration markers.")

    def combine_outcar_files(self, output_filename="OUTCAR_combined"):
        overlap_line = self.find_overlap_line()

        with open(self.original_outcar, 'r') as original_file:
            original_lines = original_file.readlines()

        with open(self.restarted_outcar, 'r') as restarted_file:
            restarted_lines = restarted_file.readlines()

        # Combine lines from the original file with the non-overlapping part of the restarted file
        combined_lines = original_lines + restarted_lines[overlap_line:]

        # Save the combined lines to a new file
        with open(output_filename, 'w') as combined_file:
            combined_file.writelines(combined_lines)
        
        self.combined_outcar = output_filename
        print(f"Combined OUTCAR file saved as {output_filename}")
    
    def get_combined_filename(self):
        return self.combined_outcar

# Example usage:
# combiner = OutcarCombiner("OUTCAR_original", "OUTCAR_restarted")
# combiner.combine_outcar_files("OUTCAR_combined")

import re
class LammpsInputFileHandler():
    def __init__(self, template_file, output_file, parameters=None):
        """
        Initializes the LammpsInputFileHandler with a template file and output file.

        Args:
            template_file (str): Path to the template file.
            output_file (str): Path to the output file.
        """
        self.template_file = template_file
        self.output_file = output_file
        self.parameters = parameters
        print(f'input file handler parameters: {self.parameters} {self.template_file} {self.output_file}')

    def generate_input_file(self):
        with open(self.template_file, 'r') as file:
            file_content = file.read()

        for placeholder, value in self.parameters.items():
            file_content = file_content.replace(f'{{{placeholder}}}', str(value))

        with open(self.output_file, 'w') as file:
            file.write(file_content)

class LammpsInputFileModifier:
    def __init__(self, input_file):
        """
        Initializes the LammpsInputFileModifier with an input file.

        Args:
            input_file (str): Path to the input file.
        """
        self.input_file = input_file
        self.parameters = {}

    def modify_input_file(self, parameters=None):
        print(f'modifying input file: {self.input_file} with parameters: {parameters}')
        if parameters is None:
            return
        with open(self.input_file, 'r') as file:
            lines = file.readlines()
        modified_lines = []
        for line in lines:
            modified = False
            for pattern, params in parameters.items():
                base_pattern = pattern.split(" {parameter}")[0]
                if base_pattern in line:
                    regex_pattern = re.escape(base_pattern) + r"\s+(.+)"
                    match = re.match(regex_pattern, line.strip())
                    if match:
                        existing_params = match.group(1).split()
                        new_params = params + existing_params[len(params):]
                        new_line = base_pattern + " " + " ".join(new_params) + "\n"
                        modified_lines.append(new_line)
                        modified = True
                        break
            if not modified:
                modified_lines.append(line)
        with open(self.input_file, 'w') as file:
            file.writelines(modified_lines)


        
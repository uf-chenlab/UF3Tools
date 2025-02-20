class LammpsInputFileHandler():
    def __init__(self, template_file, output_file, **parameters):
        """
        Initializes the LammpsInputFileHandler with a template file and output file.

        Args:
            template_file (str): Path to the template file.
            output_file (str): Path to the output file.
        """
        self.template_file = template_file
        self.output_file = output_file
        self.parameters = parameters
        print(f'input file handler parameters: {self.parameters}')

    def generate_input_file(self):
        with open(self.template_file, 'r') as file:
            file_content = file.read()

        for placeholder, value in self.parameters.items():
            file_content = file_content.replace(f'{{{placeholder}}}', str(value))

        with open(self.output_file, 'w') as file:
            file.write(file_content)


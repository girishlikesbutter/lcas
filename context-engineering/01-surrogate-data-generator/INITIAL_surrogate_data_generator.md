## FEATURE:

Create a boilerplate and class structure for a new script called 'surrogate_data_generator.py' that will generate training data for surrogate models. The script should use argparse for command-line arguments and define a DataGenerator class that loads satellite models using the existing ModelConfigManager and generates synthetic data samples.

The DataGenerator class should:
- Accept a config path in its __init__ method
- Load the satellite model using ModelConfigManager and load_satellite_from_yaml
- Store the loaded model as an instance variable
- Have a placeholder method run_generation(self, num_samples, output_path) for the main generation logic

## EXAMPLES:

The following files provide patterns to follow:
- `generate_lightcurves_pytorch.py`: Shows how to use argparse, load models with ModelConfigManager, and structure a generation script
- `src/config/model_config.py`: Shows the ModelConfigManager class and how model configurations are loaded

Key patterns to extract:
- Command-line argument parsing with argparse
- Model loading using ModelConfigManager
- Project path handling and imports
- Error handling and user feedback

## DOCUMENTATION:

- The ModelConfigManager documentation in src/config/model_config.py
- The argparse module documentation: https://docs.python.org/3/library/argparse.html
- The existing model loading patterns in generate_lightcurves_pytorch.py

## OTHER CONSIDERATIONS:

- The script should follow the existing project structure and import patterns
- Use absolute imports and proper path handling like in generate_lightcurves_pytorch.py
- Include proper docstrings following the Google style used in the project
- The script should be executable from the command line with: `python surrogate_data_generator.py --config data/models/my_model_config.yaml --samples 10000 --output data/training/is901_data.csv`
- Include helpful error messages if required arguments are missing
- The placeholder run_generation method should just print a message indicating it would generate the requested number of samples
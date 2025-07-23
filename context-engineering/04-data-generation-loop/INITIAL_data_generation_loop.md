## FEATURE:

Implement the data storage and progress logic for the DataGenerator class in surrogate_data_generator.py. The run_generation method should:
- Use sample_uniform_sphere to generate uniformly distributed sun vectors
- Iterate through these vectors with tqdm progress tracking
- Call _calculate_lit_fractions_for_sun_vector (which needs to be created) for each vector
- Collect input sun vectors and output lit fractions dictionaries
- Create a pandas DataFrame from the collected data
- Save to CSV with proper header: sun_vec_x,sun_vec_y,sun_vec_z,face_1_name,face_2_name,...
- Ensure face column order matches the dictionary order

## EXAMPLES:

- surrogate_data_generator.py - Contains the DataGenerator class with placeholder run_generation method and existing calculate_lit_fractions method
- generate_lightcurves_pytorch.py - Shows CSV generation pattern using numpy.savetxt with column stacking and header formatting
- src/pytorch/pytorch_shadow_engine.py - Contains shadow calculation patterns that could be adapted for _calculate_lit_fractions_for_sun_vector

## DOCUMENTATION:

- pandas DataFrame documentation: https://pandas.pydata.org/docs/reference/frame.html
- tqdm progress bar documentation: https://tqdm.github.io/
- numpy-quaternion documentation: https://quaternion.readthedocs.io/en/latest/

## OTHER CONSIDERATIONS:

- The project currently doesn't use pandas or tqdm, so these need to be added to requirements.txt
- The existing calculate_lit_fractions method takes a single sun vector and returns lit fractions for all conceptual faces
- Need to create _calculate_lit_fractions_for_sun_vector as a wrapper that calls calculate_lit_fractions
- The CSV should have consistent face column ordering across all rows
- Face names come from component.conceptual_faces_map dictionaries
- The method should handle cases where different components may have different conceptual faces
- Progress bar should show meaningful information (e.g., "Processing sun vectors: X/Y")
- Consider memory efficiency when collecting data for large sample sizes
- The existing logging infrastructure should be used to report progress milestones
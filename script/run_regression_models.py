import build_model

variables_for_gapfilling = {
                            'finseflux': ['ET_filtered'],
                            'adventdalen': ['ET_filtered','co2_flux_filtered'], 
                            'iskoras': ['ET_filtered'],
                            'myr1': ['ET_filtered'],
                            'myr2': ['ET_filtered']
                            }

build_model.random_forest_regression(project_id = 'finseflux_database',
                variables_for_gapfilling = variables_for_gapfilling['finseflux'])

build_model.random_forest_regression(project_id = 'myr1_database',
                variables_for_gapfilling = variables_for_gapfilling['myr1'])

build_model.random_forest_regression(project_id = 'myr2_database',
                variables_for_gapfilling = variables_for_gapfilling['myr2'])

build_model.random_forest_regression(project_id = 'iskoras_database',
                variables_for_gapfilling = variables_for_gapfilling['iskoras'])

build_model.random_forest_regression(project_id = 'adventdalen_database',
                variables_for_gapfilling = variables_for_gapfilling['adventdalen'])

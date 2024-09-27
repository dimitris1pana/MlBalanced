# Create an instance of the pipeline
pipeline = DataPipeline()

# Load data
source = '/path/to/your/database.sqlite'  # Replace with your database path
query = 'SELECT * FROM your_table WHERE your_conditions'  # Replace with your SQL query
pipeline.load_data(source=source, query=query, file_type='sqlite')

# Define target columns and balance criteria
target_columns = ['your_target_column']  # Replace with your target column(s)

# Define balance criteria as a dictionary of labels and their corresponding conditions
balance_criteria = {
    1: lambda df: (df['feature1'] >= threshold1) & (df['feature2'] < threshold2),
    0: lambda df: (df['feature1'] < threshold1) | (df['feature2'] >= threshold2)
}

# Preprocess data
pipeline.preprocess_data(target_columns=target_columns, balance_criteria=balance_criteria)

# Split data into training and testing sets
pipeline.split_data(test_size=0.1)

# Scale data
pipeline.scale_data()

# Build model
input_shape = pipeline.X_train.shape[1]
pipeline.build_model(input_shape=input_shape)

# Train model
pipeline.train_model(epochs=100, batch_size=32, patience=10)

# Evaluate model
pipeline.evaluate_model()

# Save model and scaler
pipeline.save_model(model_path='path/to/save/model.h5', scaler_path='path/to/save/scaler.pkl')

# Plot model architecture
pipeline.plot_model(plot_path='path/to/save/model_plot.png')

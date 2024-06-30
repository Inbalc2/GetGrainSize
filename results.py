import statistics

# Sample list of numeric values
data = [
    41.14, 34.81, 37, 38.79, 34.09, 32.09, 32.78, 35.47, 49.17, 37.21, 
    42.13, 40.2, 48.42, 38.21, 39.79, 57.5, 49.98, 53.03, 56.03, 43.76, 
    36.66, 46.71, 37.75, 44.67, 40.58, 31.38, 33.62, 32.15, 41.33, 34.07, 
    35.09, 46.81, 45.67, 39.7, 65.06, 48.54, 30.35, 37.36, 60.59, 47.79, 
    36.11, 36.16, 52.88, 35.67, 44.73, 36.35, 39.64, 42.03, 36.46, 52.13, 
    38.58, 47.26, 40.72, 87.84, 39.85, 34.04, 49.32, 47.95, 38.88, 34.71, 
    46.07, 46.74, 31.61, 39.66, 48.01, 37.43, 34.02, 36.39, 37.46, 51.47, 
    37.49, 46.95, 63.67, 35.93, 38.1, 47.83, 42.84, 47.96, 34.53, 42.87
]


# Compute statistics
mean_value = statistics.mean(data)
median_value = statistics.median(data)
mode_value = statistics.mode(data)
standard_deviation = statistics.stdev(data)
variance = statistics.variance(data)
min_value = min(data)
max_value = max(data)
sum_value = sum(data)

# Print statistics
print("GT:", mean_value)
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
print("Standard Deviation:", standard_deviation)
print("Variance:", variance)
print("Minimum:", min_value)
print("Maximum:", max_value)
print("Sum:", sum_value)
print("-----------------------------------")

# Sample list of numeric values
data = [
    58.53, 42.56, 31.59, 41.84, 41.71, 38.97, 29.61, 37.72, 52.29, 40.49, 
    43.03, 30.29, 28.44, 49.53, 32.65, 24.25, 53.73, 67.12, 45.15, 29.86, 
    48.46, 40.38, 29.11, 36.84, 33.2, 61.4, 36.14, 36.27, 28.35, 38.42, 
    32.46, 36.34, 33.45, 39.45, 35.66, 34.73, 47.85, 30.28, 42.97, 48.61, 
    49.48, 41.45, 40.14, 31.63, 30.08, 33.44, 36.89, 30.31, 65.57, 34.21, 
    45, 54.15, 29.01, 39.05, 42.33, 36.83, 35.99, 36.25, 42.77, 42.46, 
    27.84, 28.52, 35.72, 42.34, 45.55, 43.44, 37.71, 62.05, 29.97, 39.13, 
    30.29, 42.1, 79.7, 32.95, 36.96, 33.59, 37.94, 33.79, 34.44, 35.47, 
    32.89, 46.44, 41.39, 53.33, 34.41, 30.54, 47.42, 33.17, 36.53, 44.98, 
    41.8, 27.48, 39.4, 41.18, 38.57, 46.07, 46.3, 28.57, 40.1, 34.46, 
    32.73, 37.46, 27.49, 28.42, 84.96, 55.73, 37.39, 35.49, 33.13, 39.28, 
    26.15, 39.16, 42.22, 30.99, 47.42, 37.57, 29.95, 36.74, 34.57, 73.67, 
    41.93, 48.05, 36.81, 35.84, 41.23, 37.53, 43.36, 39.57
]


# Compute statistics
mean_value = statistics.mean(data)
median_value = statistics.median(data)
mode_value = statistics.mode(data)
standard_deviation = statistics.stdev(data)
variance = statistics.variance(data)
min_value = min(data)
max_value = max(data)
sum_value = sum(data)

# Print statistics
print("Mlography:", mean_value)
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
print("Standard Deviation:", standard_deviation)
print("Variance:", variance)
print("Minimum:", min_value)
print("Maximum:", max_value)
print("Sum:", sum_value)
print("-----------------------------------")


# Sample list of numeric values
data = [
    51.29, 36.57, 36.23, 43.02, 44.52, 33.22, 30.85, 34.02, 53.3, 36.51,
    41.48, 30.79, 33.52, 56.3, 32.87, 32.16, 41.07, 36.53, 30.71, 32.17,
    43.07, 33.97, 32.55, 33.93, 34.54, 33.81, 43.14, 41.74, 34.25, 47.3,
    34.19, 36.73, 31.81, 30.87, 37.69, 47.32, 56.77, 34.33, 36.55, 41.73,
    36.56, 36.68, 60.66, 45.97, 33.47, 61.74, 74.27, 32.95, 33.86, 29.21,
    35.29, 64.47, 42.84, 36.75, 30.29, 33.3, 31.58, 30.92, 43.29, 42.73,
    31.18, 32.64, 33.41, 31.94, 44.47, 38.17, 32.69, 44.84, 43.55, 37.04,
    33.86, 35.35, 40.39, 29.93, 48.17, 33.3, 62.47, 31.66, 37.15, 31.49,
    35.53, 36.59, 40.56, 42, 56.14, 48.22, 45.38, 32.29, 41.98, 39.37,
    39.98, 37.65, 42.57, 43.94, 34.92, 41.57, 41.2, 38.31, 28.87, 30.73,
    44.54, 36.81, 27.39, 34.71, 53.19, 41.37, 32.64, 44.33, 36.48, 55.83,
    27.51, 35.09, 33.17, 28.27, 39.61, 39.4, 35.56, 35.83, 36.7, 55.86,
    40.94, 39.09, 31.49, 44.86, 39.75, 40.84, 39.43, 48.91
]



# Compute statistics
mean_value = statistics.mean(data)
median_value = statistics.median(data)
mode_value = statistics.mode(data)
standard_deviation = statistics.stdev(data)
variance = statistics.variance(data)
min_value = min(data)
max_value = max(data)
sum_value = sum(data)

# Print statistics
print("Clemex:", mean_value)
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
print("Standard Deviation:", standard_deviation)
print("Variance:", variance)
print("Minimum:", min_value)
print("Maximum:", max_value)
print("Sum:", sum_value)
print("-----------------------------------")


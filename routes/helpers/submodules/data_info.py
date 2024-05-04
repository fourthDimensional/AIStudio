import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('robotics_match_classification.csv')

profile = ProfileReport(df, title="Pandas Profiling Report")

profile.to_file("your_report.html")
# As a JSON string
json_data = profile.to_json()

# As a file
profile.to_file("your_report.json")

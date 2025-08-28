def get_elements_in_feature_data(self, feature_data):
        columns = feature_data.columns
        elements = set()
        for col in columns:
            if col.startswith('n_'):
                element = col.split('_')[1]
                elements.add(element)
                
def daily_millis():
    now = datetime.now()
    return (now.hour * 3600 + now.minute * 60 + now.second) * 1000 + int(now.microsecond / 1000)
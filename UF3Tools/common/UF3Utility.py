def get_elements_in_feature_data(self, feature_data):
        columns = feature_data.columns
        elements = set()
        for col in columns:
            if col.startswith('n_'):
                element = col.split('_')[1]
                elements.add(element)
import pandas as pd
from openprompt.data_utils import InputExample

class CMCSDatasetHandler:
    """
    Handles loading and preprocessing for Code-Mixed Cross-Script (CMCS) data.
    Supports Kannada-English code-mixed datasets.
    """
    def __init__(self):
        # Mapping labels to integers
        self.label_map = {"neutral": 0, "hate": 1}
    
    def get_examples(self, data_list):
        """
        Converts a list of dicts/tuples into OpenPrompt InputExamples.
        """
        examples = []
        for i, item in enumerate(data_list):
            text = item.get('text', "")
            label = self.label_map.get(item.get('label', "neutral"), 0)
            
            example = InputExample(guid=i, text_a=text, label=label)
            examples.append(example)
        return examples

    def load_from_file(self, file_path):
        """
        Dynamically loads data from a CSV or Excel file.
        Expected columns: 'text', 'label'
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv, .xls, or .xlsx")
        
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
        data_list = df.to_dict('records')
        return self.get_examples(data_list)

    def get_mock_data(self):
        """
        Generates small mock data for Kannada-English CMCS evaluation.
        """
        return [
            {"text": "ಯಾರು ಫಿದಾ ಆದ್ರಿ like maadi", "label": "neutral"},
            {"text": "Worst song ever, delete this", "label": "hate"},
            {"text": "Superb acting bro, keep it up", "label": "neutral"},
            {"text": "ನೀನು ಬಹಳ ಕೆಟ್ಟವನು", "label": "hate"},
            {"text": "Amazing music and lyrics", "label": "neutral"}
        ]

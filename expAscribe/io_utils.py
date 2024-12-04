import pandas as pd
import numpy as np


def array_to_gml(arr:np.ndarray, file_name:str, labels:list[str]):
    with open(f"{file_name}", "w") as f:
        f.write("graph [\n")
        f.write("  directed 1\n")  
        
        for i in range(arr.shape[0]):
            f.write(f"  node [\n    id {i}\n    label \"{labels[i]}\"\n  ]\n")

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] != 0:
                    f.write(f"  edge [\n    source {i}\n    target {j}\n  ]\n")
        
        f.write("]\n")



def gml_to_array(file_name: str):
    with open(file_name, "r") as f:
        lines = f.readlines()

    nodes = []
    edges = []

    for line in lines:
        line = line.strip()
        if line.startswith("node"):
            current_node = None
        elif line.startswith("id"):
            current_node = int(line.split()[-1])
            nodes.append(current_node)
        elif line.startswith("edge"):
            source = None
            target = None
        elif line.startswith("source"):
            source = int(line.split()[-1])
        elif line.startswith("target"):
            target = int(line.split()[-1])
            edges.append((source, target))

    n = max(nodes) + 1 
    adjacency_matrix = np.zeros((n, n), dtype=int)

    for source, target in edges:
        adjacency_matrix[source, target] = 1

    return adjacency_matrix



def load_geo_series_matrix(file_path:str) -> pd.DataFrame:

    data = []
    header = []
    sample_title_map = {}
    sample_geo_accession = []

    def remove_quotes(s):
        return s.replace('"', '').strip()

    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_start_idx = None
    data_end_idx = None
    sample_title_idx = None
    sample_geo_accession_idx = None

    for idx, line in enumerate(lines):
        if line.strip() == "!series_matrix_table_begin":
            data_start_idx = idx + 1
        elif line.strip() == "!series_matrix_table_end":
            data_end_idx = idx
            break
        elif line.strip().startswith("!Sample_title"):
            sample_title_idx = idx
        elif line.strip().startswith("!Sample_geo_accession"):
            sample_geo_accession_idx = idx

    if data_start_idx is None or data_end_idx is None:
        raise ValueError("illegal file format")

    if sample_title_idx is not None and sample_geo_accession_idx is not None:
        sample_title_line = lines[sample_title_idx].strip()
        sample_geo_accession_line = lines[sample_geo_accession_idx].strip()
        
        sample_titles = [remove_quotes(title) for title in sample_title_line.split('\t')[1:]]  
        sample_accessions = [remove_quotes(acc) for acc in sample_geo_accession_line.split('\t')[1:]] 

        sample_title_map = dict(zip(sample_accessions, sample_titles))

    header_line = lines[data_start_idx].strip()
    header = header_line.split('\t')

    data_lines = lines[data_start_idx + 1:data_end_idx]

    for line in data_lines:
        data.append(line.strip().split('\t'))

    df = pd.DataFrame(data, columns=header)
    
    if "\"ID_REF\"" in df.columns:
        df = df.drop(columns=["\"ID_REF\""])

    df.columns = [sample_title_map.get(remove_quotes(col), col) for col in df.columns]

    return df



def load_geo_series_matrix_beta(uploaded_file) -> pd.DataFrame:
    data = []
    header = []
    sample_title_map = {}
    sample_geo_accession = []

    def remove_quotes(s):
        return s.replace('"', '').strip()

    # Check if a file was uploaded
    if uploaded_file is not None:
        # Read the uploaded file into a list of lines
        lines = uploaded_file.read().decode('utf-8').splitlines()

        data_start_idx = None
        data_end_idx = None
        sample_title_idx = None
        sample_geo_accession_idx = None

        for idx, line in enumerate(lines):
            if line.strip() == "!series_matrix_table_begin":
                data_start_idx = idx + 1
            elif line.strip() == "!series_matrix_table_end":
                data_end_idx = idx
                break
            elif line.strip().startswith("!Sample_title"):
                sample_title_idx = idx
            elif line.strip().startswith("!Sample_geo_accession"):
                sample_geo_accession_idx = idx

        if data_start_idx is None or data_end_idx is None:
            raise ValueError("Illegal file format")

        if sample_title_idx is not None and sample_geo_accession_idx is not None:
            sample_title_line = lines[sample_title_idx].strip()
            sample_geo_accession_line = lines[sample_geo_accession_idx].strip()

            sample_titles = [remove_quotes(title) for title in sample_title_line.split('\t')[1:]]
            sample_accessions = [remove_quotes(acc) for acc in sample_geo_accession_line.split('\t')[1:]]

            sample_title_map = dict(zip(sample_accessions, sample_titles))

        header_line = lines[data_start_idx].strip()
        header = header_line.split('\t')

        data_lines = lines[data_start_idx + 1:data_end_idx]

        for line in data_lines:
            data.append(line.strip().split('\t'))

        df = pd.DataFrame(data, columns=header)

        # Drop the "ID_REF" column if it exists
        if "\"ID_REF\"" in df.columns:
            df = df.drop(columns=["\"ID_REF\""])

        # Rename the columns using the sample_title_map
        df.columns = [sample_title_map.get(remove_quotes(col), col) for col in df.columns]

        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no file is uploaded



def from_arrayexpress_load_emtab_data(file:str) -> pd.DataFrame:
    # Read the txt file into a DataFrame
    df = pd.read_csv(file, sep='\t')
    #pd.readcsv supports StringIO, BytesIO and path 
    def is_serial_number(series):
        return np.all(series.diff() == 1)

    # Iterate over each column
    for col in df.columns:
        # Convert the column name to lowercase
        col_lower = col.lower()
        # Check if the column name contains "id", "number", or "name"
        if 'id' in col_lower or 'number' in col_lower or 'name' in col_lower or 'ref' in col_lower:
            # Drop the column from the DataFrame
            df = df.drop(col, axis=1)
        # Check if the data type of the column is not numeric
        elif df[col].dtype not in ['int64', 'float64']:
            # Drop the column from the DataFrame
            df = df.drop(col, axis=1)
        elif is_serial_number(df[col]):
            # Drop the column if it's similar to a serial number
            df = df.drop(col, axis=1)
    
    return df



def load_data(file:str,type:str) -> pd.DataFrame:
    if type == "csv":
        return pd.read_csv(file)
    elif type == "gse":
        return load_geo_series_matrix(file)
    elif type == "e-mtab":
        return from_arrayexpress_load_emtab_data(file)


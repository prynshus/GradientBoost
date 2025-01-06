import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xls_file = r"Dataset/DatasetXLS/train.xls"

data = pd.read_excel(xls_file)

csv_file = r"D:\Project\GradientBoost\Dataset\DatasetCSV\train.csv"
data.to_csv(csv_file,index=False)

xls_file = r"Dataset/DatasetXLS/test.xls"

data = pd.read_excel(xls_file)

csv_file = r"D:\Project\GradientBoost\Dataset\DatasetCSV\test.csv"
data.to_csv(csv_file,index=False)
import os.path

import pandas as pd
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    labels_filepath = "../static/Ford files.xlsx"
    labels = pd.read_excel(labels_filepath)
    labels = labels.dropna(how="all").reset_index(drop=True)
    labels.columns = labels.iloc[0, :]
    labels = labels.drop([0], axis=0)

    # check if all images from labels exist
    for ind, row in labels.iterrows():
        filename = row["FileName"]
        # reading image
        if os.path.exists(f"../static/ford/train/{filename}.jpg"):
            image_path = f"../static/ford/train/{filename}.jpg"
        elif os.path.exists(f"../static/ford/test/{filename}.jpg"):
            image_path = f"../static/ford/test/{filename}.jpg"
        else:
            image_path = None
            print(filename)

        if image_path:
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            # displaying image
            # plt.show()

    pass

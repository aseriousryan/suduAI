import numpy as np
import pdf2image
import cv2 #OpenCV library for python
from scipy.spatial import distance
import pytesseract
import csv
import pandas as pd
from PIL import Image
import re
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def convert_pdf_to_image(document, dpi):
    images = []
    images.extend(
                    list(
                        map(
                            lambda image: cv2.cvtColor(
                                np.asarray(image), code=cv2.COLOR_RGB2BGR
                            ),
                            pdf2image.convert_from_path(document, dpi=dpi),
                        )
                    )
                )
    return images[0]

def create_borders(img, filename):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    edges = cv2.Canny(gray, 50,150,apertureSize = 3)

    # Detect main three horizontal lines
    minLineLength = 100 # Min length of line. Line segments shorter than this are rejected.
    maxLineGap = 100 # Maximum allowed gap between line segments to treat them as single line.
    lines = cv2.HoughLinesP(edges,1,np.pi/180,200, minLineLength,maxLineGap)
    # Filter and keep the three longest lines
    if lines is not None:
        # Calculate line lengths and sort the lines by length
        line_lengths = [distance.euclidean((x1, y1), (x2, y2)) for line in lines for x1, y1, x2, y2 in line]
        sorted_lines = [line for line, length in sorted(zip(lines, line_lengths), key=lambda x: x[1], reverse=True)]

        # Keep only the three longest lines
        final_lines = sorted_lines[:3]

        # Draw the three longest lines on the original image
        for line in final_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    sorted_lines = sorted(final_lines, key=lambda line: line[0][1])
    
    y1 = sorted_lines[0][0][1]
    y2 = sorted_lines[2][0][1]
    x1 =  sorted_lines[0][0][0]
    x2 = sorted_lines[2][0][2]

    img = img[y1:y2,x1:x2]


    df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME).dropna()

    month_pattern = re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}", flags=re.IGNORECASE)

    # Extract rows with valid month formats
    df["month_match"] = df["text"].str.match(month_pattern)
    filtered_df = df[df["month_match"] == True]
    #print(filtered_df)
    # Create a dictionary mapping each month to its left position
    month_positions = dict(zip(filtered_df["text"], filtered_df["left"] + filtered_df["width"] + 5))

    # Draw Vertical lines based on headers (month patterns)
    for month in month_positions.keys():
        left = month_positions.get(month, None)
        if left is not None:
            cv2.line(img, (left, 0), (left, img.shape[0]), (0, 0, 0), 1)

    # Find the leftmost header among the specified headers (month patterns)
    leftmost_header = filtered_df["left"].min()    

    # Draw vertical line before the leftmost header (month pattern)
    line_x = leftmost_header - 15
    cv2.line(img, (line_x, 0), (line_x, img.shape[0]), (0, 0, 0), 1)

    # Find the rightmost month in the DataFrame
    rightmost_month_idx = (filtered_df["left"] + filtered_df["width"]).idxmax()
    rightmost_month = filtered_df.loc[rightmost_month_idx]

   # rightmost_month_row = df[df["text"] == rightmost_month].iloc[0]

    text_below_dec = df[(df["left"] >= rightmost_month["left"]) & (df["left"] <= (rightmost_month["left"] + rightmost_month["width"]))]

    # Drawing horizontal lines above text elements below the rightmost month
    for _, row in text_below_dec.iloc[1:].iterrows():
        cv2.line(img, (0, row["top"] - 20), (img.shape[1], row["top"] - 20), (0, 0, 0), 1)  # Horizontal lines above text


    cv2.imwrite(f"{filename}.jpg", img)

def extract_table(file, filename):
    #read your file
    img = cv2.imread(file,0)
    img.shape

    #thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #inverting the image 
    img_bin = 255-img_bin

    # countcol(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)


    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)


    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)


    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    #Get mean of heights
    mean = np.mean(heights)

    #Create list box to store all boxes in  
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])
            


    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0

    #Sorting the boxes to their respective row and column
    for i in range(len(box)):    
            
        if(i==0):
            column.append(box[i])
            previous=box[i]    
        
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]            
                
                if(i==len(box)-1):
                    row.append(column)        
                
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])
                


    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]

    center=np.array(center)
    center.sort()
    #Regarding the distance to the columns center, the boxes are arranged in respective order

    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)


    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=''
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=2)
                    
                    out = pytesseract.image_to_string(erosion)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner +" "+ out
                outer.append(inner)

    #Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    dataframe.replace('\n', ' ', regex=True, inplace=True)

    dataframe = dataframe[~dataframe.astype(str).apply(lambda x: x.str.contains(r'\*', case=False)).any(axis=1)]
    
    # Reset the index
    dataframe = dataframe.reset_index(drop=True)    

    
    # Set the header to the second row
    dataframe.columns = dataframe.iloc[0]
    dataframe.columns = dataframe.columns.str.strip()
    dataframe = dataframe.rename(columns={col: col.replace(' ', '') for col in dataframe.columns})
    dataframe.drop("Total", axis=1, inplace=True)
    
    dataframe.replace(to_replace=r'/', value='7', regex=True, inplace=True)



    # Drop the second row as it's now the header
    dataframe = dataframe.drop(0)

    # Reset the index
    dataframe = dataframe.reset_index(drop=True)

    if 'Debtor' in dataframe.columns:
        # Debtor code
        for i in range(len(dataframe)-1, 0, -1):
            if dataframe.iloc[i, 1:].isna().all():
                dataframe.at[i - 1, 'Debtor'] += ' ' + dataframe.at[i, 'Debtor']
                dataframe = dataframe.drop(index=i)

        df = dataframe.reset_index(drop=True)
        df = df.set_index('Debtor')
        df = df.T
        date = pd.to_datetime(df.index, format='%b-%Y')
        df['Year'] = date.year
        df['Month'] = date.month_name()
        df = df.reset_index(drop=True)

        for col in df.columns:
            if df[col].dtype == object:  # Check only object (string) columns
                if df[col].str.contains(',').any():  # Check if column contains commas
                    df[col] = df[col].replace(',', '', regex=True).apply(pd.to_numeric)

    else:
        # Creditor code
        for i in range(len(dataframe)-1, 0, -1):
            if dataframe.iloc[i, 1:].isna().all():
                dataframe.at[i - 1, 'Creditor'] += ' ' + dataframe.at[i, 'Creditor']
                dataframe = dataframe.drop(index=i)

        df = dataframe.reset_index(drop=True)
        df = df.set_index('Creditor')
        df = df.T
        date = pd.to_datetime(df.index, format='%b-%Y')
        df['Year'] = date.year
        df['Month'] = date.month_name()
        df = df.reset_index(drop=True)

        for col in df.columns:
            if df[col].dtype == object:  # Check only object (string) columns
                if df[col].str.contains(',').any():  # Check if column contains commas
                    df[col] = df[col].replace(',', '', regex=True).apply(pd.to_numeric)


    df.to_csv(f"{filename}.csv")

    return dataframe



if __name__ == '__main__':
    image = convert_pdf_to_image('C:/Users/Anish/Desktop/SuduAI/suduAI/input_data/AP_MONTHLY.pdf', 500)
    create_borders(image, "test")
    extract_table("test.jpg", "test")
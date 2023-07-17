import pandas as pd
import csv

## DATA PREPROCESSING
# Load the annotations into a DataFrame
df = pd.read_csv('./Dataset/Train_Tagged_Titles.tsv', delimiter='\t', header=0, quoting=csv.QUOTE_NONE)
df.columns = ['Record Number', 'Title', 'Token', 'Tag']

# Add columns
df['Aspect Value'] = df['Token']
df['Aspect Name'] = df['Tag']
df['Entity'] = df['Tag'].isnull()


# Forward fill the null tag values with the previous non-null tag value
df['Aspect Name'].fillna(method='ffill', inplace=True)

# loop through the rows in IsNull using iterrows()
sameEntity = 0
outer_index = 0
while outer_index < len(df) - 1:
    skipIncrement = False

    # Get the Aspect Name
    aspectName = df.at[outer_index, 'Aspect Name']

    # Check if it is not part of an entity
    if df.at[outer_index + 1, 'Entity'] != True:
        df.at[outer_index, 'Entity'] = sameEntity
    else:
        inner_index = outer_index
        while df.at[inner_index, 'Aspect Name'] == aspectName:
            df.at[inner_index, 'Entity'] = sameEntity
            inner_index += 1
        outer_index = inner_index
        skipIncrement = True

    sameEntity += 1
    if skipIncrement == False:
        outer_index += 1

#Handle last edge case
df.at[outer_index, 'Entity'] = sameEntity

# Convert 'Aspect Value' column to strings
df['Aspect Value'] = df['Aspect Value'].astype(str)

# Delete rows with the following Values in the tag column
df = df.drop(df[df['Tag'] == 'No Tag'].index)
df = df.drop(df[df['Tag'] == 'Obscure'].index)

# Delete columns
df.drop(columns=['Title'],inplace=True)
df.drop(columns='Token',inplace=True)
df.drop(columns='Tag',inplace=True)

#group by Record Number, IsNull and Aspect Value
df = df.groupby(['Record Number', 'Entity', 'Aspect Name'])['Aspect Value'].apply(lambda x: ' '.join(x)).reset_index()

# Delete column
df = df.drop(columns=['Entity'])

# Set the display option to show all columns
pd.set_option('display.max_columns', None)

# Print the modified DataFrame
#print(df.head(20))




import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3


def load_data(messages_filepath, categories_filepath):
    """This function loads the two datasets, merges them into one dataframe and returns the dataframe as df """
    # Read the first dataset
    messages = pd.read_csv(messages_filepath)
    # Read the second dataset
    categories = pd.read_csv(categories_filepath)
    # Merge the two datasets using the common id
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    # Split the values in the categories column on the ; character so that each value becomes a separate column.
    categories = df['categories'].str.split(pat=';', expand=True)
    # Use the first row of categories dataframe to create column names for the categories data.
    row = categories.iloc[0]
    first_extraction = []
    for i in row:
        m = i.split('-')
        first_extraction.append(m[0])
    category_colnames = first_extraction
    # or simply categories[column].apply(lambda x: x.split('-')[1]
    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x.strip()[-1] for x in categories[column]]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    all_duplicates = df[df.duplicated(keep=False)]
    df = df.drop_duplicates()
    # If we want the number of dropped duplicates
    number_of_dropped_duplicates = len(df) - len(df.drop_duplicates())
    # Also we will drop na values, because we won't need it in our further analysis
    df = df.dropna(subset=['related'], axis=0)
    # After inspecting the categories columns, 'related' has three values = [0,1,2]
    # We will convert the values 2 into values 1
    df[df['related'] == 2] = 1
    # Also, we can drop the column 'child alone', because it doesn't provide any important information
    # because the values are all zero
    # So, we can not use it for training, since the model won't know when to classify into that category
    df = df.drop(['child_alone'], axis=1)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Disaster_messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

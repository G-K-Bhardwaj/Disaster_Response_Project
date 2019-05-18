# import required libraries
import sys
import pandas as pd
import sqlalchemy as sqla

# create function to load data from messages file and categories file
def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        messages_filepath (str): messages csv files path
        categories_filepath (str): categories csv file path 
    OUTPUT: 
        df: dataframe having messages and cateries details

    DESCRIPTION:
            read messages csv file as messages dataframe and 
            categories csv file as categories dataframe
            merge both the dataframes as df applying inner join on ['id'] column 
    '''

    # read data files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets.
    # - Merge the messages and categories datasets using the common id
    # - Assign this combined dataset to `df`, which will be cleaned in the following steps
    
    # merge datasets
    df = pd.merge(messages, categories, how='inner', on='id')
    
    return df    



def clean_data(df):
    '''
    INPUT:
        df (pandas dataframe): dataframe having messages and their belonging categories details
    OUTPUT: 
        df (pandas dataframe): cleansed dataframe

    DESCRIPTION:
            clean data in df dataframe 
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames


    # Convert category values to just numbers 0 or 1.
    # - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0`   
    # becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int32')
    

    # Replace `categories` column in `df` with new category columns.
    # - Drop the categories column from the df dataframe since it is no longer needed.
    # - Concatenate df and categories data frames.

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    INPUT:
        df (pandas dataframe): cleansed dataframe having messages and their belonging categories details
    OUTPUT: 
        df (pandas dataframe): database having Messages table

    DESCRIPTION:
            save dataframe as Messages table in database file as provide as input   
    '''
    table = 'Messages'

    engine = sqla.create_engine('sqlite:///'+database_filename)
    
    # drop table if exists
    connection = engine.raw_connection()
    cursor = connection.cursor()
    command = "DROP TABLE IF EXISTS {};".format(table)
    cursor.execute(command)
    connection.commit()
    cursor.close()

    df.to_sql(name=table, con=engine, index=False)  


def main():
    if len(sys.argv) == 4:

        # get user arguments
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # load data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # cleanse data
        print('Cleaning data...')
        df = clean_data(df)
        
        # save cleansed data
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

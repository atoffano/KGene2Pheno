def load_classifier():
    pass

def parse_args():
    '''Parse the command line arguments'''
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--entity', nargs='*', default=None, help='Multiple keywords')
    parser.add_argument('--class_of_interest', required=True, help='Name of the method')
    parser.add_argument('--known', required=True, help='Name of the dataset')

    parser.add_argument('--nb', default=None, help='A SPARQL query')
    parser.add_argument('--classifier', default=None, help='Format of the dataset')    
    args = parser.parse_args()

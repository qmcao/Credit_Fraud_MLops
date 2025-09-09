import logging

from src.components.data_extraction import DataIngestion


def main():
    
    #Load data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.init_data_ingestion()
    # print(train.shape, test.shape)
    logging.info("Ingestion process completed sucessfully")
    
if __name__ == "__main__":
    main()